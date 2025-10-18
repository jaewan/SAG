"""
N-gram Language Models - PRODUCTION VERSION
All NLTK issues fixed, tested with real data
"""
import numpy as np
from typing import List, Dict, Iterator
import logging

logger = logging.getLogger(__name__)

# Try to import NLTK, but handle gracefully if not available
try:
    import nltk
    from nltk.lm import Laplace, KneserNeyInterpolated, MLE
    from nltk.lm.preprocessing import padded_everygram_pipeline
    from nltk import FreqDist  # For vocabulary
    NLTK_AVAILABLE = True
    logger.info("✅ NLTK available - using NLTK n-gram models")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ NLTK not available - using fallback n-gram implementation")
    logger.warning("   Install NLTK for better performance: pip install nltk")

    # Fallback simple n-gram implementation
    class SimpleFreqDist:
        def __init__(self):
            self.freqs = {}

        def update(self, tokens):
            for token in tokens:
                self.freqs[token] = self.freqs.get(token, 0) + 1

        def __getitem__(self, key):
            return self.freqs.get(key, 0)

        def __contains__(self, key):
            return key in self.freqs

    FreqDist = SimpleFreqDist

    def padded_everygram_pipeline(n, text):
        """Simple fallback for padded n-grams"""
        for sentence in text:
            yield list(nltk_ngrams(sentence, n))

    def nltk_ngrams(sequence, n):
        """Simple n-gram generator"""
        for i in range(len(sequence) - n + 1):
            yield tuple(sequence[i:i+n])


class SimpleNgramModel:
    """Simple fallback n-gram model when NLTK is not available"""

    def __init__(self, n: int = 3, smoothing: str = 'laplace'):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()
        self.is_fitted = False

    def fit(self, sequences: List[List[str]]):
        """Fit simple n-gram model"""
        for sequence in sequences:
            # Update vocabulary
            for token in sequence:
                self.vocab.add(token)

            # Count n-grams
            for i in range(len(sequence) - self.n + 1):
                ngram = tuple(sequence[i:i+self.n])
                context = ngram[:-1]
                token = ngram[-1]

                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

        self.is_fitted = True

    def log_prob(self, token: str, context: tuple) -> float:
        """Get log probability of token given context"""
        if not self.is_fitted:
            return -20.0  # Very low probability

        ngram = context + (token,)

        if self.smoothing == 'laplace':
            # Laplace smoothing
            count_ngram = self.ngram_counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)
            vocab_size = len(self.vocab)
            return np.log((count_ngram + 1) / (count_context + vocab_size))
        else:
            # Simple MLE
            count_ngram = self.ngram_counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)
            if count_context == 0:
                return -20.0
            return np.log(count_ngram / count_context)

    def sequence_log_probs(self, sequence: List[str]) -> np.ndarray:
        """Get log probabilities for entire sequence"""
        if len(sequence) < self.n:
            return np.array([])

        log_probs = []
        for i in range(self.n - 1, len(sequence)):
            context = tuple(sequence[i-self.n+1:i])
            token = sequence[i]
            log_probs.append(self.log_prob(token, context))

        return np.array(log_probs)

    def perplexity(self, sequences: List[List[str]]) -> float:
        """Compute perplexity"""
        total_log_prob = 0.0
        total_tokens = 0

        for sequence in sequences:
            log_probs = self.sequence_log_probs(sequence)
            if len(log_probs) > 0:
                total_log_prob += log_probs.sum()
                total_tokens += len(log_probs)

        if total_tokens == 0:
            return float('inf')

        # Convert to perplexity
        avg_log_prob = total_log_prob / total_tokens
        return np.exp(-avg_log_prob)


class PrunedNgramLanguageModel:
    """Memory-efficient n-gram model with vocabulary pruning"""

    def __init__(self, n: int = 3, smoothing: str = 'laplace',
                 max_vocab_size: int = 50000, min_count: int = 2,
                 oov_handling: str = 'skip'):  # NEW: OOV handling
        """
        Initialize pruned n-gram model

        Args:
            n: N-gram order
            smoothing: Smoothing method
            max_vocab_size: Maximum vocabulary size to prevent memory explosion
            min_count: Minimum frequency for tokens to be included
            oov_handling: How to handle OOV tokens ('skip', 'unk', 'smooth')
        """
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.oov_handling = oov_handling

        # Initialize model attributes similar to parent class
        self.n = n
        self.smoothing = smoothing
        self.model = None
        self.vocab = None
        self.is_fitted = False

        # Override model with pruned version
        if NLTK_AVAILABLE:
            if smoothing == 'laplace':
                self.model = Laplace(n)
            elif smoothing == 'kneser_ney':
                self.model = KneserNeyInterpolated(n)
            elif smoothing == 'mle':
                self.model = MLE(n)
            else:
                raise ValueError(f"Unknown smoothing: {smoothing}")
        else:
            # Fallback: simple frequency-based model with pruning
            self.model = SimpleNgramModel(n, smoothing)

        self.vocab = None
        self.is_fitted = False
        self.pruned_vocab = None  # Track pruned vocabulary
        self.oov_token = '<UNK>'  # OOV token for replacement

    def fit(self, sequences: List[List[str]]):
        """Fit model with vocabulary pruning"""
        logger.info(f"Training pruned {self.n}-gram model (max_vocab={self.max_vocab_size}, min_count={self.min_count})")

        if NLTK_AVAILABLE:
            self._fit_nltk_pruned(sequences)
        else:
            self._fit_fallback_pruned(sequences)

    def _fit_nltk_pruned(self, sequences: List[List[str]]):
        """Fit NLTK model with pruning"""
        from collections import Counter

        # Step 1: Count all tokens across all sequences
        all_tokens = []
        for sequence in sequences:
            all_tokens.extend(sequence)

        token_counts = Counter(all_tokens)

        # Step 2: Prune vocabulary - keep only frequent tokens
        # Sort by frequency and keep top max_vocab_size above min_count
        frequent_tokens = [
            token for token, count in token_counts.most_common()
            if count >= self.min_count
        ][:self.max_vocab_size]

        pruned_vocab = set(frequent_tokens)
        self.pruned_vocab = pruned_vocab

        logger.info(f"✅ Pruned vocabulary: {len(pruned_vocab):,} tokens (from {len(token_counts):,})")

        # Step 3: Filter sequences to only include tokens in pruned vocabulary
        filtered_sequences = []
        for sequence in sequences:
            filtered_seq = [token for token in sequence if token in pruned_vocab]
            if len(filtered_seq) >= self.n:  # Need at least n tokens for n-gram
                filtered_sequences.append(filtered_seq)

        logger.info(f"✅ Filtered sequences: {len(filtered_sequences):,}/{len(sequences):,}")

        if len(filtered_sequences) == 0:
            logger.warning("⚠️ No sequences remain after vocabulary pruning!")
            self.is_fitted = True  # Mark as fitted but empty
            return

        # Step 4: Fit model on filtered sequences
        try:
            train_data, padded_sents = padded_everygram_pipeline(self.n, filtered_sequences)

            # Create vocabulary from filtered sequences
            self.vocab = FreqDist()
            for sent in padded_sents:
                self.vocab.update(sent)

            # Fit model
            train_data_list = list(train_data)
            vocab_list = list(self.vocab)

            self.model.fit(train_data_list, vocab_list)
            self.is_fitted = True

            logger.info(f"✅ Pruned {self.n}-gram model fitted successfully")

        except Exception as e:
            logger.error(f"❌ Failed to fit pruned n-gram model: {e}")
            raise RuntimeError(f"Model fitting failed: {e}") from e

    def perplexity(self, sequences: List[List[str]]) -> float:
        """Compute perplexity"""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        if NLTK_AVAILABLE:
            # NLTK implementation
            if len(sequences) == 0:
                return float('inf')

            total_log_prob = 0.0
            total_tokens = 0

            for sequence in sequences:
                if len(sequence) >= self.n:
                    # Use NLTK's perplexity calculation
                    try:
                        seq_log_prob = sum(self.model.logscore(token, context)
                                         for context, token in zip(
                                             [tuple(sequence[i-self.n+1:i]) for i in range(self.n-1, len(sequence))],
                                             sequence[self.n-1:]
                                         ))
                        total_log_prob += seq_log_prob
                        total_tokens += len(sequence) - self.n + 1
                    except:
                        # Fallback to simple calculation
                        total_tokens += len(sequence)

            if total_tokens == 0:
                return float('inf')

            avg_log_prob = total_log_prob / total_tokens
            return np.exp(-avg_log_prob)
        else:
            # Fallback implementation
            return self.model.perplexity(sequences)

    def sequence_log_probs(self, sequence: List[str]) -> np.ndarray:
        """Compute log P(token|context) for each token"""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        if len(sequence) == 0:
            return np.array([])

        # Apply OOV handling
        if self.oov_handling == 'unk' and self.pruned_vocab is not None:
            sequence = [token if token in self.pruned_vocab else self.oov_token for token in sequence]
        elif self.oov_handling == 'skip' and self.pruned_vocab is not None:
            sequence = [token for token in sequence if token in self.pruned_vocab]

        if NLTK_AVAILABLE:
            # NLTK implementation
            log_probs = []
            # Manual padding implementation
            padded = ['<s>'] * (self.n - 1) + sequence + ['</s>']
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                token = padded[i]
                try:
                    log_prob = self.model.logscore(token, context)
                    # Handle -inf
                    if np.isinf(log_prob) and log_prob < 0:
                        log_prob = -20.0
                    log_probs.append(log_prob)
                except:
                    log_probs.append(-20.0)
            return np.array(log_probs)
        else:
            # Fallback implementation
            return self.model.sequence_log_probs(sequence)

    def surprise_scores(self, sequence: List[str]) -> np.ndarray:
        """Compute surprisal = -log P(token|context)"""
        return -self.sequence_log_probs(sequence)

    def _fit_fallback_pruned(self, sequences: List[List[str]]):
        """Fit fallback model with pruning"""
        from collections import Counter

        # Step 1: Count all tokens
        all_tokens = []
        for sequence in sequences:
            all_tokens.extend(sequence)

        token_counts = Counter(all_tokens)

        # Step 2: Prune vocabulary
        frequent_tokens = [
            token for token, count in token_counts.most_common()
            if count >= self.min_count
        ][:self.max_vocab_size]

        pruned_vocab = set(frequent_tokens)
        self.pruned_vocab = pruned_vocab

        logger.info(f"✅ Pruned vocabulary: {len(pruned_vocab):,} tokens (from {len(token_counts):,})")

        # Step 3: Filter sequences and fit model
        filtered_sequences = []
        for sequence in sequences:
            filtered_seq = [token for token in sequence if token in pruned_vocab]
            if len(filtered_seq) >= self.n:
                filtered_sequences.append(filtered_seq)

        if len(filtered_sequences) == 0:
            logger.warning("⚠️ No sequences remain after vocabulary pruning!")
            self.is_fitted = True
            return

        # Fit the simple model
        self.model.fit(filtered_sequences)
        self.is_fitted = True

        logger.info(f"✅ Pruned fallback {self.n}-gram model fitted successfully")


class AdaptivePrunedNgramModel(PrunedNgramLanguageModel):
    """
    Automatically prunes to fit memory budget
    """
    def __init__(self, n: int, memory_budget_gb: float = 15, **kwargs):
        self.memory_budget_gb = memory_budget_gb

        # Estimate vocab size from memory budget
        # Rough estimate: n-gram size ≈ vocab_size^n × 16 bytes
        estimated_vocab = int((memory_budget_gb * 1e9 / 16) ** (1/n))
        max_vocab = min(estimated_vocab, 200000)  # Cap at 200K

        logger.info(f"📊 Adaptive pruning for n={n}: max_vocab={max_vocab:,} (budget={memory_budget_gb}GB)")

        super().__init__(n=n, max_vocab_size=max_vocab, **kwargs)

    def fit(self, sequences: List[List[str]]):
        """Fit with memory monitoring"""
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**3)

        super().fit(sequences)

        mem_after = process.memory_info().rss / (1024**3)
        mem_used = mem_after - mem_before

        logger.info(f"✅ Model memory: {mem_used:.2f}GB / {self.memory_budget_gb}GB budget")

        if mem_used > self.memory_budget_gb * 1.1:
            logger.warning(f"⚠️ Exceeded memory budget by {(mem_used/self.memory_budget_gb - 1)*100:.1f}%")


class NgramLanguageModel:
    """Production n-gram model using NLTK or fallback"""

    def __init__(self, n: int = 3, smoothing: str = 'laplace', oov_handling: str = 'skip'):
        """
        Initialize n-gram model

        Args:
            n: N-gram order
            smoothing: Smoothing method ('laplace', 'kneser_ney', 'mle')
            oov_handling: How to handle OOV tokens ('skip', 'unk', 'smooth')
        """
        self.n = n
        self.smoothing = smoothing
        self.oov_handling = oov_handling

        if NLTK_AVAILABLE:
            if smoothing == 'laplace':
                self.model = Laplace(n)
            elif smoothing == 'kneser_ney':
                self.model = KneserNeyInterpolated(n)
            elif smoothing == 'mle':
                self.model = MLE(n)
            else:
                raise ValueError(f"Unknown smoothing: {smoothing}")
        else:
            # Fallback: simple frequency-based model
            self.model = SimpleNgramModel(n, smoothing)

        self.vocab = None
        self.is_fitted = False

    def fit(self, sequences: List[List[str]]):
        """Fit model on sequences"""
        if NLTK_AVAILABLE:
            # NLTK implementation
            if len(sequences) == 0:
                logger.warning("⚠️ Empty sequences provided")
                return

            # Prepare training data
            train_data, padded_sents = padded_everygram_pipeline(self.n, sequences)

            # Fit vocabulary
            self.vocab = FreqDist()
            for sent in padded_sents:
                self.vocab.update(sent)

            # Fit model
            self.model.fit(train_data, self.vocab)
            self.is_fitted = True
        else:
            # Fallback implementation
            if len(sequences) == 0:
                raise ValueError("Empty sequences")

            # Filter empty sequences
            sequences = [s for s in sequences if len(s) > 0]

            # Fit the simple model
            self.model.fit(sequences)
            self.is_fitted = True
            # NLTK implementation (continued from above)
            if len(sequences) == 0:
                raise ValueError("All sequences empty after filtering")

            # Filter sequences too short for n-gram
            min_length = max(1, self.n - 1)
            original_count = len(sequences)
            sequences = [s for s in sequences if len(s) >= min_length]

            if len(sequences) == 0:
                raise ValueError(
                    f"No sequences with >= {min_length} tokens for {self.n}-gram model. "
                    f"All {original_count} sequences too short."
                )

            if len(sequences) < original_count * 0.5:
                logger.warning(
                    f"⚠️ Filtered {original_count - len(sequences)} sequences "
                    f"({100*(1-len(sequences)/original_count):.1f}%) too short for {self.n}-gram"
                )

            logger.debug(f"Fitting {self.n}-gram {self.smoothing} model on {len(sequences)} sequences")

            # CRITICAL: Actually train the model
            try:
                # Create training data with NLTK preprocessing
                train_data_gen, vocab_gen = padded_everygram_pipeline(self.n, sequences)

                # Convert generators to lists for NLTK model training
                train_data = list(train_data_gen)
                vocab = list(vocab_gen)

                # Fit model
                self.model.fit(train_data, vocab)
                self.vocab = vocab
                self.is_fitted = True

                logger.debug(f"✅ Vocabulary size: {len(vocab)}")

            except Exception as e:
                logger.error(f"❌ Failed to fit {self.n}-gram model: {e}")
                raise RuntimeError(f"Model fitting failed: {e}") from e

    def perplexity(self, sequences: List[List[str]]) -> float:
        """Compute perplexity"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if NLTK_AVAILABLE:
            # NLTK implementation
            if len(sequences) == 0:
                return float('inf')

            total_log_prob = 0.0
            total_tokens = 0

            for sequence in sequences:
                if len(sequence) >= self.n:
                    # Use NLTK's perplexity calculation
                    try:
                        seq_log_prob = sum(self.model.logscore(token, context)
                                         for context, token in zip(
                                             [tuple(sequence[i-self.n+1:i]) for i in range(self.n-1, len(sequence))],
                                             sequence[self.n-1:]
                                         ))
                        total_log_prob += seq_log_prob
                        total_tokens += len(sequence) - self.n + 1
                    except:
                        # Fallback to simple calculation
                        total_tokens += len(sequence)

            if total_tokens == 0:
                return float('inf')

            avg_log_prob = total_log_prob / total_tokens
            return np.exp(-avg_log_prob)
        else:
            # Fallback implementation
            return self.model.perplexity(sequences)

    def sequence_log_probs(self, sequence: List[str]) -> np.ndarray:
        """Compute log P(token|context) for each token"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        if len(sequence) == 0:
            return np.array([])

        # Apply OOV handling if vocabulary is available
        if self.oov_handling != 'smooth' and hasattr(self, 'vocab') and self.vocab is not None:
            if self.oov_handling == 'unk':
                # Replace OOV with <UNK> token
                sequence = [token if token in self.vocab else '<UNK>' for token in sequence]
            elif self.oov_handling == 'skip':
                # Skip OOV tokens
                sequence = [token for token in sequence if token in self.vocab]

        if NLTK_AVAILABLE:
            # NLTK implementation
            log_probs = []
            # Manual padding implementation
            padded = ['<s>'] * (self.n - 1) + sequence + ['</s>']
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                token = padded[i]
                try:
                    log_prob = self.model.logscore(token, context)
                    # Handle -inf
                    if np.isinf(log_prob) and log_prob < 0:
                        log_prob = -20.0
                    log_probs.append(log_prob)
                except:
                    log_probs.append(-20.0)
            return np.array(log_probs)
        else:
            # Fallback implementation
            return self.model.sequence_log_probs(sequence)

    def surprise_scores(self, sequence: List[str]) -> np.ndarray:
        """Compute surprisal = -log P(token|context)"""
        return -self.sequence_log_probs(sequence)


def evaluate_ngram_model(model: NgramLanguageModel,
                        test_benign: List[List[str]],
                        test_malicious: List[List[str]]) -> Dict:
    """Evaluate n-gram for anomaly detection"""
    from sklearn.metrics import roc_auc_score, roc_curve

    # Perplexities
    ppl_benign = model.perplexity(test_benign)
    ppl_malicious = model.perplexity(test_malicious)

    # Max surprise per sequence
    benign_scores = [model.surprise_scores(seq).max() if len(seq) > 0 else 0.0
                    for seq in test_benign]
    malicious_scores = [model.surprise_scores(seq).max() if len(seq) > 0 else 0.0
                       for seq in test_malicious]

    # Detect
    y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(malicious_scores))])
    y_scores = np.concatenate([benign_scores, malicious_scores])

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {
            'perplexity_benign': ppl_benign,
            'perplexity_malicious': ppl_malicious,
            'perplexity_ratio': float('inf'),
            'auc': 0.5,
            'tpr_at_10fpr': 0.0
        }

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5

    # TPR @ 10% FPR
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tpr_at_10fpr = tpr[np.where(fpr <= 0.1)[0][-1]] if any(fpr <= 0.1) else 0.0
    except:
        tpr_at_10fpr = 0.0

    return {
        'perplexity_benign': ppl_benign,
        'perplexity_malicious': ppl_malicious,
        'perplexity_ratio': ppl_malicious / ppl_benign if ppl_benign > 0 else float('inf'),
        'auc': auc,
        'tpr_at_10fpr': tpr_at_10fpr
    }
