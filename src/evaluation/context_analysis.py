"""
Context Analysis - PRODUCTION VERSION
CRITICAL FIX: Hybrid CV strategy that works with sparse attacks
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import gc  # ✅ ADDED for garbage collection
import psutil  # ✅ ADDED for memory monitoring
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ContextWindowAnalyzer:
    """
    Context window analysis with ROBUST CV
    KEY FIX: Handles sparse attacks gracefully
    """

    def __init__(self,
                 n_values: List[int] = None,
                 cv_folds: int = 5,
                 min_malicious_per_fold: int = 2,
                 cache_tokenization: bool = True):
        self.n_values = n_values or [1, 2, 3, 5, 10, 25]
        self.cv_folds = cv_folds
        self.min_malicious_per_fold = min_malicious_per_fold
        self.cache_tokenization = cache_tokenization
        self._token_cache = {}  # ✅ ADDED: Simple tokenization cache

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

        # Run
        results = {}
        for n in self.n_values:
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

        # ✅ ADDED: Clear tokenization cache to free memory
        if self.cache_tokenization:
            cache_size = len(self._token_cache)
            self._token_cache.clear()
            logger.info(f"🗑️ Cleared tokenization cache ({cache_size} entries)")

        return results, decision

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

        # Decision logic
        if n_mal_users < self.cv_folds:
            # Too few attackers for GroupKFold
            logger.info(f" → Using STRATIFIED CV (few attackers)")
            return "stratified"
        elif n_mal_users >= self.cv_folds * 2:
            # Enough attackers
            logger.info(f" → Using GROUPED CV (prevents leakage)")
            return "grouped"
        else:
            # Borderline - try both
            logger.info(f" → Using HYBRID CV (try both, use best)")
            return "hybrid"

    def _tokenize(self, session: Dict) -> List[str]:
        """
        ENHANCED tokenization with user/computer context

        Format: <auth_type>_<outcome>_<user_type>_<host_pattern>
        Example: "Kerberos_Success_regular_single"
        """
        # ✅ ADDED: Simple caching for tokenization
        if self.cache_tokenization:
            # Create cache key from session characteristics
            cache_key = (
                session['user_id'],
                len(session['events']),
                tuple(sorted(set(e['dst_computer'] for e in session['events'])))[:5]  # First 5 hosts
            )

            if cache_key in self._token_cache:
                return self._token_cache[cache_key]

        tokens = []

        # Infer user type (admin/regular/service) - handle both string and int user_ids
        user_id = session['user_id']
        if isinstance(user_id, str):
            if "ANONYMOUS LOGON" in user_id:
                user_type = "service"
            elif "@" in user_id:
                user_type = "regular"  # Domain user
            else:
                user_type = "regular"  # Local user
        else:
            # Handle integer user_ids (legacy format)
            if user_id < 100:
                user_type = "admin"
            elif 10000 <= user_id < 11000:
                user_type = "service"
            else:
                user_type = "regular"

        # Host diversity (single vs multi-host)
        unique_hosts = len(set(e['dst_computer'] for e in session['events']))
        if unique_hosts == 1:
            host_pattern = "single"
        elif unique_hosts <= 3:
            host_pattern = "few"
        else:
            host_pattern = "many"

        for event in session['events']:
            token = f"{event['auth_type']}_{event['outcome']}_{user_type}_{host_pattern}"
            tokens.append(token)

        # ✅ ADDED: Cache the result
        if self.cache_tokenization:
            self._token_cache[cache_key] = tokens

        return tokens

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
        """Run CV with given splitter"""
        fold_results = []

        if groups is not None:
            splits = cv.split(all_sessions, y, groups)
        else:
            splits = cv.split(all_sessions, y)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
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
                continue

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
                continue

            # Evaluate
            try:
                metrics = evaluate_ngram_model(model, test_benign_seqs, test_mal_seqs)
                fold_results.append(metrics)
                logger.info(f" AUC: {metrics['auc']:.3f}, TPR@10%: {metrics['tpr_at_10fpr']:.3f}")
            except Exception as e:
                logger.warning(f" ⚠️ Eval failed: {e}")
                # ✅ ADDED: Cleanup on failure
                del model
                gc.collect()
                continue

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


# Import here to avoid circular imports
from src.models.ngram_models import NgramLanguageModel, evaluate_ngram_model
