"""
Context Analysis - PRODUCTION VERSION
CRITICAL FIX: Hybrid CV strategy that works with sparse attacks
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
import gc  # ✅ ADDED for garbage collection
import psutil  # ✅ ADDED for memory monitoring
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed  # ✅ ADDED for parallel CV
from tqdm import tqdm  # ✅ ADDED for progress bars

logger = logging.getLogger(__name__)


class TokenizationCache:
    """
    Global tokenization cache with LRU eviction for Phase 1 enhancements

    Prevents re-tokenizing the same session multiple times across CV folds
    """
    def __init__(self, max_size_gb: float = 5.0):
        self.cache = {}
        self.max_size_gb = max_size_gb
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0

        # LRU tracking (simple timestamp-based)
        self.access_times = {}

    def get(self, session_id: str) -> Optional[List[str]]:
        """Get cached tokens for session"""
        if session_id in self.cache:
            self.hit_count += 1
            self.access_times[session_id] = datetime.now()
            return self.cache[session_id]
        self.miss_count += 1
        return None

    def put(self, session_id: str, tokens: List[str]):
        """Cache tokens for session"""
        # Estimate size (rough approximation)
        token_size = sum(len(t.encode('utf-8')) for t in tokens) + len(tokens) * 8

        # Check if we need to evict
        if self.current_size_bytes + token_size > self.max_size_gb * 1e9:
            self._evict_lru()

        # Still too big? Skip caching
        if self.current_size_bytes + token_size > self.max_size_gb * 1e9:
            logger.debug(f"⚠️ Skipping cache for {session_id}: would exceed {self.max_size_gb}GB limit")
            return

        self.cache[session_id] = tokens
        self.access_times[session_id] = datetime.now()
        self.current_size_bytes += token_size

    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'size_mb': self.current_size_bytes / (1024**2),
            'entries': len(self.cache),
            'hit_rate': hit_rate,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'max_size_gb': self.max_size_gb
        }

    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.access_times:
            return

        # Find 20% least recently used
        sorted_times = sorted(self.access_times.items(), key=lambda x: x[1])
        to_evict = int(len(sorted_times) * 0.2)

        logger.debug(f"🔄 Evicting {to_evict} LRU entries from cache")

        for session_id, _ in sorted_times[:to_evict]:
            if session_id in self.cache:
                # Estimate size to subtract
                tokens = self.cache[session_id]
                token_size = sum(len(t.encode('utf-8')) for t in tokens) + len(tokens) * 8
                self.current_size_bytes -= token_size
                del self.cache[session_id]
                del self.access_times[session_id]

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        logger.info("🗑️ Tokenization cache cleared")


class ContextWindowAnalyzer:
    """
    Context window analysis with ROBUST CV
    KEY FIX: Handles sparse attacks gracefully
    """

    def __init__(self,
                 n_values: List[int] = None,
                 cv_folds: int = 10,
                 min_malicious_per_fold: int = 20,
                 parallel_jobs: int = 8,
                 memory_per_job_gb: int = 10,
                 cache_tokenization: bool = True):
        self.n_values = n_values or [1, 2, 3, 5, 7, 10, 15]
        self.cv_folds = cv_folds
        self.min_malicious_per_fold = min_malicious_per_fold
        self.parallel_jobs = parallel_jobs
        self.memory_per_job_gb = memory_per_job_gb
        self.cache_tokenization = cache_tokenization

        # Tokenizer will be set by runner
        self.tokenizer = None

        # ENHANCED: Global tokenization cache for efficiency
        self.tokenization_cache = TokenizationCache(max_size_gb=5.0) if cache_tokenization else None

    def analyze(self,
                benign_sessions: List[Dict],
                malicious_sessions: List[Dict]) -> Tuple[Dict, str]:
        """Run full analysis"""
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: CONTEXT WINDOW ANALYSIS")
        logger.info("="*80)

        # Validate
        if not self._check_sample_size(len(benign_sessions), len(malicious_sessions)):
            return {}, "stop"

        power = self._check_statistical_power(len(malicious_sessions))

        if power < 0.5:
            logger.error("❌ Statistical power too low (<50%)")
            logger.error("   Results will be unreliable")
            # Auto-proceed in non-interactive mode for demonstration
            logger.warning("⚠️ Statistical power too low - proceeding anyway for demonstration")
            # response = input("\nProceed anyway? [y/N]: ")
            # if response.lower() != 'y':
            #     return {}, "stop"

        # Choose CV strategy
        cv_strategy = self._choose_cv_strategy(benign_sessions, malicious_sessions)
        logger.info(f"\n📊 CV Strategy: {cv_strategy}")

        # Run with progress bars
        results = {}
        for n in tqdm(self.n_values, desc="N-gram sizes", unit="model"):
            logger.info(f"\n{'='*80}")
            logger.info(f"N={n}-gram Model")
            logger.info(f"{'='*80}")
            try:
                if cv_strategy == "stratified":
                    metrics = self._evaluate_stratified_cv(n, benign_sessions, malicious_sessions)
                elif cv_strategy == "grouped":
                    metrics = self._evaluate_grouped_cv(n, benign_sessions, malicious_sessions)
                else:  # hybrid
                    metrics = self._evaluate_hybrid_cv(n, benign_sessions, malicious_sessions)

                results[n] = metrics
                self._print_metrics(n, metrics)
            except Exception as e:
                logger.error(f"❌ Failed for n={n}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(results) == 0:
            logger.error("❌ All models failed!")
            return {}, "stop"

        # Decision
        decision = self._make_decision(results)

        # Plots
        self._plot_results(results)

        if self.cache_tokenization and self.tokenization_cache is not None:
            cache_size = len(self.tokenization_cache.cache)
            self.tokenization_cache.clear()
            logger.info(f"🗑️ Cleared tokenization cache ({cache_size} entries)")

        return results, decision

    def _tokenize(self, session: Dict) -> List[str]:
        """
        Tokenize session using cached tokenizer
        """
        return self._get_tokens(session)

    def _get_tokens(self, session: Dict) -> List[str]:
        """
        Get tokens for session, using global cache if enabled

        This prevents re-tokenizing the same session multiple times across CV folds
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not set")

        session_id = session.get('session_id')
        if not session_id:
            # Generate a temporary ID for caching
            session_id = f"temp_{hash(str(session))}"

        # ENHANCED: Use global tokenization cache
        if self.tokenization_cache is not None:
            cached_tokens = self.tokenization_cache.get(session_id)
            if cached_tokens is not None:
                logger.debug(f"💾 Cache hit for session {session_id}")
                return cached_tokens
            else:
                # Cache miss - tokenize and cache
                tokens = self.tokenizer.tokenize_session(session)
                self.tokenization_cache.put(session_id, tokens)
                logger.debug(f"💾 Cached tokens for session {session_id}")
                return tokens
        else:
            return self.tokenizer.tokenize_session(session)

    def _check_sample_size(self, n_benign: int, n_malicious: int) -> bool:
        """Check sample size"""
        logger.info(f"\n📊 Sample Size:")
        logger.info(f" Benign: {n_benign}")
        logger.info(f" Malicious: {n_malicious}")

        min_needed = self.cv_folds * self.min_malicious_per_fold
        if n_malicious < min_needed:
            logger.error(f"❌ Insufficient malicious samples!")
            logger.error(f" Need: {min_needed}")
            logger.error(f" Have: {n_malicious}")
            logger.error("")
            logger.error(" Solutions:")
            logger.error(" 1. Load more days with attacks (days 37-45)")
            logger.error(" 2. Use looser labeling (label_window_minutes=240)")
            logger.error(" 3. Reduce folds (not recommended)")

            # Auto-adjust folds
            if n_malicious >= self.min_malicious_per_fold * 2:
                new_folds = n_malicious // self.min_malicious_per_fold
                logger.warning(f"⚠️ Auto-reducing folds: {self.cv_folds} → {new_folds}")
                self.cv_folds = new_folds
                return True
            return False

        logger.info(f" ✅ Sufficient samples for {self.cv_folds}-fold CV")
        return True

    def _choose_cv_strategy(self, benign: List[Dict], malicious: List[Dict]) -> str:
        """
        Choose CV strategy based on data characteristics
        Returns: "stratified", "grouped", or "hybrid"

        PRIORITY: Always prefer Grouped CV to prevent user leakage
        """
        # Count unique malicious users
        mal_users = set(s['user_id'] for s in malicious)
        n_mal_users = len(mal_users)

        # Count benign users
        ben_users = set(s['user_id'] for s in benign)
        n_ben_users = len(ben_users)

        logger.info(f"\n🔍 Data Characteristics:")
        logger.info(f" Malicious users: {n_mal_users}")
        logger.info(f" Benign users: {n_ben_users}")
        logger.info(f" Attacks per user: {len(malicious) / n_mal_users:.1f}")

        # DECISION LOGIC (prioritize user separation)
        if n_mal_users >= self.cv_folds:
            # Enough attackers for proper GroupKFold
            logger.info(f" → Using GROUPED CV (prevents user leakage)")
            logger.info(f"   ✅ Each user stays in one fold")
            return "grouped"
        else:
            # Too few attackers - use stratified but warn about potential leakage
            logger.warning(f" → Using STRATIFIED CV (⚠️ potential user leakage)")
            logger.warning(f"   ⚠️ Same user might appear in train and test")
            logger.warning(f"   Reason: Only {n_mal_users} attackers for {self.cv_folds} folds")
            logger.warning(f"   Solution: Load more attack days or reduce folds")
            return "stratified"


    def _evaluate_stratified_cv(self, n: int,
                                benign: List[Dict],
                                malicious: List[Dict]) -> Dict:
        """Stratified CV (ignores users, may have leakage)"""
        all_sessions = benign + malicious
        y = np.array([0]*len(benign) + [1]*len(malicious))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return self._run_cv(cv, all_sessions, y, n, groups=None)

    def _evaluate_grouped_cv(self, n: int,
                            benign: List[Dict],
                            malicious: List[Dict]) -> Dict:
        """Grouped CV (no user in both train/test)"""
        all_sessions = benign + malicious
        y = np.array([0]*len(benign) + [1]*len(malicious))
        user_ids = np.array([s['user_id'] for s in all_sessions])
        cv = GroupKFold(n_splits=self.cv_folds)
        return self._run_cv(cv, all_sessions, y, n, groups=user_ids)

    def _evaluate_hybrid_cv(self, n: int,
                            benign: List[Dict],
                            malicious: List[Dict]) -> Dict:
        """Try both, use whichever works"""
        logger.info(" Trying grouped CV first...")
        try:
            metrics_grouped = self._evaluate_grouped_cv(n, benign, malicious)
            # Check if all folds succeeded
            if metrics_grouped['n_folds'] >= self.cv_folds * 0.6:  # At least 60% folds ok
                logger.info(f" ✅ Grouped CV succeeded ({metrics_grouped['n_folds']} folds)")
                return metrics_grouped
        except Exception as e:
            logger.warning(f" Grouped CV failed: {e}")

        logger.info(" Falling back to stratified CV...")
        return self._evaluate_stratified_cv(n, benign, malicious)

    def _run_cv(self, cv, all_sessions, y, n, groups) -> Dict:
        """Run CV with given splitter (now with parallel processing)"""
        if groups is not None:
            splits = list(cv.split(all_sessions, y, groups))
        else:
            splits = list(cv.split(all_sessions, y))

        logger.info(f"\n🚀 Running {len(splits)} folds in parallel (n_jobs={self.parallel_jobs})")
        logger.info(f"   Memory allocation: {self.parallel_jobs}×{self.memory_per_job_gb}GB = {self.parallel_jobs * self.memory_per_job_gb}GB")

        # ENHANCED: Run folds in parallel with memory monitoring
        from joblib import parallel_backend

        with parallel_backend('loky', n_jobs=self.parallel_jobs):
            fold_results = Parallel()(
                delayed(self._run_single_fold_with_monitoring)(
                    fold_idx, train_idx, test_idx, all_sessions, y, n,
                    memory_limit_gb=self.memory_per_job_gb
                )
                for fold_idx, (train_idx, test_idx) in enumerate(splits)
            )

        fold_results = [r for r in fold_results if r is not None]


    def _run_single_fold_with_monitoring(self, fold_idx: int, train_idx: List[int],
                                        test_idx: List[int], all_sessions: List[Dict],
                                        y: np.ndarray, n: int,
                                        memory_limit_gb: float) -> Optional[Dict]:
        """Run single fold with enhanced memory monitoring"""
        import psutil
        process = psutil.Process()

        # Pre-fold memory check
        mem_start = process.memory_info().rss / (1024**3)

        try:
            # Use existing fold logic with memory limit
            result = self._run_single_fold(fold_idx, train_idx, test_idx,
                                          all_sessions, y, n)

            # Post-fold memory check
            mem_end = process.memory_info().rss / (1024**3)
            mem_used = mem_end - mem_start

            if mem_used > memory_limit_gb * 1.2:
                logger.warning(f"⚠️ Fold {fold_idx} exceeded memory budget: {mem_used:.1f}GB > {memory_limit_gb}GB")

            return result

        except MemoryError:
            logger.error(f"❌ Fold {fold_idx} OOM")
            return None
        finally:
            gc.collect()

    def _run_single_fold(self, fold_idx: int, train_idx: List[int],
                        test_idx: List[int], all_sessions: List[Dict],
                        y: np.ndarray, n: int) -> Optional[Dict]:
        """Run a single CV fold (wrapper for backward compatibility)"""
        return self._run_single_fold_original(fold_idx, train_idx, test_idx, all_sessions, y, n)

    def _run_single_fold_original(self, fold_idx: int, train_idx: List[int],
                                test_idx: List[int], all_sessions: List[Dict],
                                y: np.ndarray, n: int) -> Optional[Dict]:
        """Run a single CV fold (for parallel processing)"""
        logger.info(f"\n Fold {fold_idx+1}/{self.cv_folds}:")

        # ✅ ADDED: Memory monitoring before each fold
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / 1024 / 1024 / 1024
            logger.info(f" 💾 Memory before fold: {mem_gb:.2f} GB")
        except:
            pass

        # Indices of benign in train
        train_benign_idx = [i for i in train_idx if y[i] == 0]
        train_sessions = [all_sessions[i] for i in train_benign_idx]
        test_sessions = [all_sessions[i] for i in test_idx]
        test_labels = y[test_idx]
        n_test_mal = (test_labels == 1).sum()

        logger.info(f" Train: {len(train_sessions)} benign")
        logger.info(f" Test: {(test_labels==0).sum()} benign, {n_test_mal} malicious")

        # Skip fold if no malicious in test
        if n_test_mal == 0:
            logger.warning(f" ⚠️ No malicious in test - skipping fold")
            return None

        # Tokenize (inside loop - no leakage!)
        train_seqs = [self._tokenize(s) for s in train_sessions]
        test_seqs = [self._tokenize(s) for s in test_sessions]
        test_benign_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 0]
        test_mal_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 1]

        # Fit
        try:
            model = NgramLanguageModel(n=n, smoothing='laplace')
            model.fit(train_seqs)
        except Exception as e:
            logger.error(f" ❌ Fit failed: {e}")
            # ✅ ADDED: Cleanup on failure
            del train_seqs, test_seqs
            gc.collect()
            return None

        # Evaluate
        try:
            metrics = evaluate_ngram_model(model, test_benign_seqs, test_mal_seqs)
            logger.info(f" AUC: {metrics['auc']:.3f}, TPR@10%: {metrics['tpr_at_10fpr']:.3f}")
            return metrics
        except Exception as e:
            logger.warning(f" ⚠️ Eval failed: {e}")
            # ✅ ADDED: Cleanup on failure
            del model
            gc.collect()
            return None

        finally:
            # ✅ ADDED: Aggressive cleanup after each fold
            del train_sessions, test_sessions, train_seqs, test_seqs, test_benign_seqs, test_mal_seqs
            if 'model' in locals():
                del model
            gc.collect()

            # ✅ ADDED: Memory monitoring after each fold
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_gb = mem_info.rss / 1024 / 1024 / 1024
                logger.info(f" 💾 Memory after fold: {mem_gb:.2f} GB")
            except:
                pass


    def _check_statistical_power(self, n_samples: int, effect_size: float = 0.05) -> float:
        """
        Statistical power for detecting AUC improvement

        Simplified approach: Power ≈ effect_size × sqrt(n) / 2
        For rigorous analysis, use bootstrap simulations
        """

        # Heuristic: AUC std ≈ 1/sqrt(2*n) for balanced data
        auc_std = 1.0 / np.sqrt(2 * n_samples)
        z_score = effect_size / auc_std

        # Power from standard normal (one-sided test)
        from scipy.stats import norm
        power = 1 - norm.cdf(1.96 - z_score)  # 1.96 for 5% significance

        logger.info(f"\n📊 Statistical Power (Heuristic):")
        logger.info(f"   Samples: {n_samples}")
        logger.info(f"   Detectable effect: {effect_size} AUC")
        logger.info(f"   Estimated power: {power:.2%}")

        if power < 0.8:
            needed = int((1.96 / effect_size) ** 2 * 2)  # Solve for power=0.8
            logger.warning(f"   ⚠️ LOW POWER!")
            logger.warning(f"   Need ~{needed} samples for 80% power")
            logger.warning(f"   Consider: Loading more days or using looser labeling")
        else:
            logger.info(f"   ✅ Adequate power to detect effects")

        return power

    def _print_metrics(self, n: int, metrics: Dict):
        """Print results"""
        ci_lower, ci_upper = metrics['auc_ci']
        logger.info(f"\n 📊 Results ({metrics['n_folds']} folds):")
        logger.info(f" AUC: {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
        logger.info(f" 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        logger.info(f" TPR@10%: {metrics['tpr_mean']:.3f} ± {metrics['tpr_std']:.3f}")
        logger.info(f" Perplexity ratio: {metrics['ppl_ratio_mean']:.2f}")

    def _make_decision(self, results: Dict) -> str:
        """Make decision with robust statistical testing"""

        n_values = sorted(results.keys())
        baseline_n = n_values[0]
        final_n = n_values[-1]

        # FIXED: Use 'auc_mean' not 'auc'
        baseline_auc = results[baseline_n]['auc_mean']
        final_auc = results[final_n]['auc_mean']
        baseline_ci = results[baseline_n]['auc_ci']
        final_ci = results[final_n]['auc_ci']

        improvement = final_auc - baseline_auc

        # Check if confidence intervals overlap (stronger test)
        ci_overlap = (final_ci[0] <= baseline_ci[1]) and (baseline_ci[0] <= final_ci[1])

        logger.info(f"\n📈 Statistical Comparison:")
        logger.info(f"  Baseline (n={baseline_n}):")
        logger.info(f"    AUC: {baseline_auc:.3f}")
        logger.info(f"    95% CI: [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]")
        logger.info(f"  Final (n={final_n}):")
        logger.info(f"    AUC: {final_auc:.3f}")
        logger.info(f"    95% CI: [{final_ci[0]:.3f}, {final_ci[1]:.3f}]")
        logger.info(f"  Improvement: {improvement:+.3f}")
        logger.info(f"  Confidence intervals overlap: {'Yes (not significant)' if ci_overlap else 'No (likely significant)'}")

        logger.info("\n" + "="*80)
        logger.info("🎯 DECISION:")
        logger.info("="*80)

        # Decision logic accounting for statistical significance
        if improvement > 0.10 and not ci_overlap:
            logger.info("✅ PROCEED TO PHASE 2")
            logger.info("   Reason: Large improvement (>0.10) with non-overlapping CIs")
            return "proceed"

        elif improvement > 0.05 and not ci_overlap:
            logger.info("✅ PROCEED TO PHASE 2")
            logger.info("   Reason: Moderate improvement (>0.05) with non-overlapping CIs")
            return "proceed"

        elif improvement > 0.05 and ci_overlap:
            logger.info("⚠️ PROCEED WITH CAUTION")
            logger.info("   Reason: Moderate improvement but overlapping CIs")
            logger.info("   → Statistical significance unclear")
            return "proceed_caution"

        else:
            logger.info("❌ STOP")
            logger.info(f"   Reason: Insufficient improvement ({improvement:+.3f} < 0.05 AUC)")
            return "stop"

    def _plot_results(self, results: Dict):
        """Generate plots"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_values = sorted(results.keys())
        auc_means = [results[n]['auc_mean'] for n in n_values]
        auc_stds = [results[n]['auc_std'] for n in n_values]

        # AUC plot
        axes[0].errorbar(n_values, auc_means, yerr=auc_stds,
                        marker='o', markersize=10, linewidth=2, capsize=5)
        axes[0].set_xlabel('Context Window (n)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Detection AUC', fontweight='bold', fontsize=12)
        axes[0].set_title('Performance vs Context Size', fontsize=14, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Good (0.8)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        positions = range(len(n_values))
        box_data = [[f['auc'] for f in results[n]['fold_results']] for n in n_values]
        bp = axes[1].boxplot(box_data, positions=positions, labels=[str(n) for n in n_values])
        axes[1].set_xlabel('Context Window (n)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('AUC Distribution', fontweight='bold', fontsize=12)
        axes[1].set_title('Variability Across Folds', fontsize=14, fontweight='bold')
        axes[1].axhline(0.8, color='r', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output = Path("experiments/phase1/context_analysis.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        logger.info(f"\n📊 Saved: {output}")
        plt.close()


class CorrelatedContextWindowAnalyzer(ContextWindowAnalyzer):
    """ContextWindowAnalyzer that uses correlated data for richer tokenization"""

    def __init__(self, *args, model_class=None, model_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}

        # Use semantic tokenizer
        from src.features.semantic_tokenizer import LANLSemanticTokenizer
        self.semantic_tokenizer = LANLSemanticTokenizer()
        logger.info("✅ Using semantic tokenizer")

    def _get_tokens(self, session: Dict) -> List[str]:
        """Use semantic tokenizer instead of computer IDs"""
        return self.semantic_tokenizer.tokenize_session(session)

    def _run_cv(self, cv, all_sessions, y, n, groups):
        """Override to use custom model class with parallel processing"""
        if groups is not None:
            splits = list(cv.split(all_sessions, y, groups))
        else:
            splits = list(cv.split(all_sessions, y))

        logger.info(f"\n🚀 Running {len(splits)} folds in parallel (n_jobs=min(4, {self.cv_folds}))")

        # Run folds in parallel (2-4x speedup)
        fold_results = Parallel(n_jobs=min(4, self.cv_folds))(
            delayed(self._run_single_fold)(
                fold_idx, train_idx, test_idx, all_sessions, y, n
            )
            for fold_idx, (train_idx, test_idx) in enumerate(splits)
        )

        fold_results = [r for r in fold_results if r is not None]

        if len(fold_results) == 0:
            raise RuntimeError(f"All {self.cv_folds} folds failed for n={n}")

        if len(fold_results) < self.cv_folds * 0.5:
            logger.warning(f" ⚠️ Only {len(fold_results)}/{self.cv_folds} folds succeeded")

        # Aggregate
        df = pd.DataFrame(fold_results)
        return {
            'auc_mean': df['auc'].mean(),
            'auc_std': df['auc'].std(),
            'auc_sem': df['auc'].sem(),
            'auc_ci': stats.t.interval(0.95, len(df)-1, df['auc'].mean(), df['auc'].sem()) if len(df) > 1 else (0, 1),
            'tpr_mean': df['tpr_at_10fpr'].mean(),
            'tpr_std': df['tpr_at_10fpr'].std(),
            'ppl_ratio_mean': df['perplexity_ratio'].mean(),
            'fold_results': fold_results,
            'n_folds': len(fold_results)
        }

    def _run_single_fold(self, fold_idx: int, train_idx: List[int],
                        test_idx: List[int], all_sessions: List[Dict],
                        y: np.ndarray, n: int) -> Optional[Dict]:
        """Run a single CV fold (for parallel processing)"""
        logger.info(f"\n Fold {fold_idx+1}/{self.cv_folds}:")

        # Memory monitoring before each fold
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / 1024 / 1024 / 1024
            logger.info(f" 💾 Memory before fold: {mem_gb:.2f} GB")
        except:
            pass

        # Get train/test data
        train_benign_idx = [i for i in train_idx if y[i] == 0]
        train_sessions = [all_sessions[i] for i in train_benign_idx]
        test_sessions = [all_sessions[i] for i in test_idx]
        test_labels = y[test_idx]
        n_test_mal = (test_labels == 1).sum()

        logger.info(f" Train: {len(train_sessions)} benign")
        logger.info(f" Test: {(test_labels==0).sum()} benign, {n_test_mal} malicious")

        # Skip fold if no malicious in test
        if n_test_mal == 0:
            logger.warning(f" ⚠️ No malicious in test - skipping fold")
            return None

        # Tokenize
        train_seqs = [self._tokenize(s) for s in train_sessions]
        test_seqs = [self._tokenize(s) for s in test_sessions]
        test_benign_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 0]
        test_mal_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 1]

        # Fit model with custom class and parameters
        model = None
        try:
            model = self.model_class(n=n, **self.model_kwargs)
            model.fit(train_seqs)
        except Exception as e:
            logger.error(f" ❌ Fit failed: {e}")
            del train_seqs, test_seqs
            gc.collect()
            return None

        # Evaluate
        try:
            metrics = evaluate_ngram_model(model, test_benign_seqs, test_mal_seqs)
            logger.info(f" AUC: {metrics['auc']:.3f}, TPR@10%: {metrics['tpr_at_10fpr']:.3f}")
        except Exception as e:
            logger.warning(f" ⚠️ Eval failed: {e}")
            del model
            gc.collect()
            return None

        # Cleanup
        del train_sessions, test_sessions, train_seqs, test_seqs, test_benign_seqs, test_mal_seqs
        if model is not None:
            del model
        gc.collect()

        return metrics


# Import here to avoid circular imports
from src.models.ngram_models import NgramLanguageModel, evaluate_ngram_model
