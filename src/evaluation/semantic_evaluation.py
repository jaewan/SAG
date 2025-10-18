"""
Semantic Disambiguation Evaluation

This is THE KEY TEST for justifying SAG:
- Can models distinguish rare-benign from rare-malicious?
- Standard metrics (AUC, F1) don't capture this!
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


def cohen_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


class SemanticEvaluator:
    """
    Evaluate semantic disambiguation capability

    Key Questions:
    1. Can model distinguish admin-at-3AM (benign) from employee-at-3AM (malicious)?
    2. What's precision on rare-but-benign events?
    3. What's recall on semantically-similar attacks?
    """

    def evaluate(self, model, test_sessions: List[Dict],
                 tokenize_fn) -> Dict:
        """
        Run comprehensive semantic evaluation

        Args:
            model: Trained n-gram model
            test_sessions: List of test session dicts
            tokenize_fn: Function to tokenize sessions

        Returns:
            Dict with semantic metrics
        """
        logger.info("\n" + "="*80)
        logger.info("SEMANTIC DISAMBIGUATION EVALUATION")
        logger.info("="*80)

        # Step 1: Identify rare-but-benign cases
        rare_benign = self._find_rare_benign(test_sessions)
        logger.info(f"\n🔍 Rare Benign Cases: {len(rare_benign)}")
        if rare_benign:
            self._print_samples("Rare Benign", rare_benign[:3], tokenize_fn)

        # Step 2: Identify similar attacks
        similar_attacks = self._find_similar_attacks(test_sessions)
        logger.info(f"\n🔍 Similar Attack Cases: {len(similar_attacks)}")
        if similar_attacks:
            self._print_samples("Attack", similar_attacks[:3], tokenize_fn)

        # Step 3: Check if we have enough samples
        if len(rare_benign) < 5 or len(similar_attacks) < 5:
            logger.warning("⚠️ Insufficient samples for semantic test")
            logger.warning(f"   Need: 5+ of each")
            logger.warning(f"   Have: {len(rare_benign)} benign, {len(similar_attacks)} attacks")
            return {'status': 'insufficient_data'}

        # Step 4: Compute surprise scores
        logger.info(f"\n📊 Computing surprise scores...")
        rare_benign_scores = self._compute_scores(model, rare_benign, tokenize_fn)
        attack_scores = self._compute_scores(model, similar_attacks, tokenize_fn)

        logger.info(f"   Rare benign: mean={np.mean(rare_benign_scores):.2f}, "
                   f"median={np.median(rare_benign_scores):.2f}")
        logger.info(f"   Attacks: mean={np.mean(attack_scores):.2f}, "
                   f"median={np.median(attack_scores):.2f}")

        # Step 5: Statistical significance test
        stat_test = self._statistical_test(rare_benign_scores, attack_scores)

        # Step 6: Classification performance
        classification = self._classification_test(rare_benign_scores, attack_scores)

        # Step 7: Overall assessment
        assessment = self._overall_assessment(stat_test, classification)

        # Combine results
        results = {
            'n_rare_benign': len(rare_benign),
            'n_attacks': len(similar_attacks),
            'rare_benign_scores': {
                'mean': np.mean(rare_benign_scores),
                'median': np.median(rare_benign_scores),
                'std': np.std(rare_benign_scores)
            },
            'attack_scores': {
                'mean': np.mean(attack_scores),
                'median': np.median(attack_scores),
                'std': np.std(attack_scores)
            },
            'statistical_test': stat_test,
            'classification': classification,
            'assessment': assessment,
            'status': 'complete'
        }

        self._print_results(results)

        return results

    def _find_rare_benign(self, sessions: List[Dict]) -> List[Dict]:
        """
        Find rare but benign sessions

        Criteria:
        - Not malicious
        - Admin user OR unusual time OR unusual pattern
        - Substantial activity (5+ events)
        """
        rare_benign = []

        for session in sessions:
            if session.get('is_malicious', False):
                continue  # Skip malicious

            user_id = str(session.get('user_id', '')).upper()
            start_time = session.get('start_time')
            num_events = len(session.get('events', []))

            # Criteria 1: Admin users doing unusual activities
            is_admin = any(x in user_id for x in ['ADMIN', 'SYSTEM', 'U12', 'U23'])

            # Criteria 2: Unusual time
            is_unusual_time = False
            if start_time:
                hour = start_time.hour
                # Check if weekend (Saturday=5, Sunday=6)
                is_weekend = start_time.weekday() >= 5
                is_unusual_time = (
                    (hour < 6 or hour >= 22) or  # Night
                    is_weekend  # Weekend
                )

            # Criteria 3: High activity
            is_substantial = num_events >= 5

            # Select if:
            # - Admin user at unusual time (admin maintenance at night)
            # - OR regular user at unusual time with substantial activity
            if is_substantial and (
                (is_admin and is_unusual_time) or  # Admin at unusual time
                (not is_admin and is_unusual_time)  # Regular user at unusual time
            ):
                rare_benign.append(session)

        return rare_benign

    def _find_similar_attacks(self, sessions: List[Dict]) -> List[Dict]:
        """
        Find attacks that are statistically similar to rare benign

        Criteria:
        - Is malicious
        - Regular user (not admin)
        - Unusual time OR unusual pattern
        - Substantial activity (5+ events)
        """
        similar_attacks = []

        for session in sessions:
            if not session.get('is_malicious', False):
                continue  # Need malicious

            user_id = str(session.get('user_id', '')).upper()
            start_time = session.get('start_time')
            num_events = len(session.get('events', []))

            # Must be regular user (not admin/service)
            # Admin patterns: U1-U99, ADMIN*, SYSTEM*, *ADMIN, *SYSTEM
            is_admin_user = (
                any(x in user_id for x in ['ADMIN', 'SYSTEM']) or
                (user_id.startswith('U') and len(user_id) >= 2 and user_id[1:].isdigit() and int(user_id[1:]) <= 99)
            )
            is_regular = not is_admin_user and not any(x in user_id for x in ['SERVICE', 'ANONYMOUS'])

            # Unusual time
            is_unusual_time = False
            if start_time:
                hour = start_time.hour
                # Check if weekend (Saturday=5, Sunday=6)
                is_weekend = start_time.weekday() >= 5
                is_unusual_time = (hour < 6 or hour >= 22) or is_weekend

            # Substantial activity
            is_substantial = num_events >= 5

            if is_regular and is_unusual_time and is_substantial:
                similar_attacks.append(session)

        return similar_attacks

    def _compute_scores(self, model, sessions: List[Dict],
                       tokenize_fn) -> List[float]:
        """Compute surprise scores for sessions"""
        scores = []

        for session in sessions:
            tokens = tokenize_fn(session)
            if len(tokens) == 0:
                continue

            try:
                surprise = model.surprise_scores(tokens)
                # Use mean surprise as session score
                session_score = surprise.mean() if len(surprise) > 0 else 0.0
                scores.append(session_score)
            except:
                scores.append(0.0)

        return scores

    def _statistical_test(self, benign_scores: List[float],
                         attack_scores: List[float]) -> Dict:
        """
        Statistical test: Are distributions significantly different?
        """
        logger.info(f"\n📊 Statistical Significance Test:")

        # Mann-Whitney U test (non-parametric)
        # H0: Distributions are the same
        # H1: Attack scores are higher (one-sided)
        stat, pval = mannwhitneyu(benign_scores, attack_scores,
                                  alternative='less')  # benign < attack

        # Effect size (Cohen's d)
        effect = cohen_d(benign_scores, attack_scores)

        is_significant = pval < 0.05

        logger.info(f"   Mann-Whitney U: statistic={stat:.2f}, p-value={pval:.4f}")
        logger.info(f"   Effect size (Cohen's d): {effect:.3f}")
        logger.info(f"   Interpretation: {'Small' if abs(effect) < 0.5 else 'Medium' if abs(effect) < 0.8 else 'Large'}")

        if is_significant:
            logger.info(f"   ✅ Statistically significant (p < 0.05)")
        else:
            logger.warning(f"   ⚠️ NOT statistically significant (p >= 0.05)")

        return {
            'u_statistic': stat,
            'p_value': pval,
            'is_significant': is_significant,
            'effect_size': effect,
            'effect_interpretation': 'small' if abs(effect) < 0.5 else 'medium' if abs(effect) < 0.8 else 'large'
        }

    def _classification_test(self, benign_scores: List[float],
                            attack_scores: List[float]) -> Dict:
        """
        Classification test: Can we separate them?
        """
        logger.info(f"\n📊 Classification Performance:")

        # Create labels
        y_true = np.concatenate([
            np.zeros(len(benign_scores)),
            np.ones(len(attack_scores))
        ])
        y_scores = np.concatenate([benign_scores, attack_scores])

        # AUC
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5

        logger.info(f"   AUC: {auc:.3f}")

        # Classification at median threshold
        threshold = np.median(np.concatenate([benign_scores, attack_scores]))
        y_pred = (y_scores >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # Separate precision for benign (important!)
        benign_correct = np.sum((np.array(benign_scores) < threshold))
        benign_precision = benign_correct / len(benign_scores)

        logger.info(f"   Precision (overall): {precision:.3f}")
        logger.info(f"   Recall (attacks): {recall:.3f}")
        logger.info(f"   F1: {f1:.3f}")
        logger.info(f"   ⭐ Benign Precision: {benign_precision:.3f}")
        logger.info(f"      (% of rare-benign correctly labeled as benign)")

        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'benign_precision': benign_precision,
            'threshold': threshold
        }

    def _overall_assessment(self, stat_test: Dict, classification: Dict) -> Dict:
        """
        Overall assessment: Can model do semantic disambiguation?
        """
        logger.info(f"\n" + "="*80)
        logger.info("SEMANTIC DISAMBIGUATION ASSESSMENT")
        logger.info("="*80)

        # Criteria for "good" semantic understanding
        criteria = {
            'statistically_significant': stat_test['is_significant'],
            'large_effect_size': abs(stat_test['effect_size']) > 0.5,
            'good_auc': classification['auc'] > 0.75,
            'high_benign_precision': classification['benign_precision'] > 0.7
        }

        passed = sum(criteria.values())
        total = len(criteria)

        logger.info(f"\n✅ Criteria Met: {passed}/{total}")
        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            logger.info(f"   {status} {criterion}")

        # Overall verdict
        if passed >= 3:
            verdict = "GOOD"
            interpretation = "Model shows semantic understanding"
        elif passed >= 2:
            verdict = "PARTIAL"
            interpretation = "Model has some semantic capability but limited"
        else:
            verdict = "POOR"
            interpretation = "Model struggles with semantic disambiguation"

        # For SAG justification - add to interpretation when verdict is POOR
        if verdict == "POOR":
            interpretation += "\n💡 SAG Justification:\n   ✅ N-grams fail at semantic disambiguation\n   ✅ This proves the semantic gap exists\n   ✅ SAG's symbolic guidance is needed!"

        logger.info(f"\n🎯 Verdict: {verdict}")
        logger.info(f"   {interpretation}")

        return {
            'criteria': criteria,
            'passed': passed,
            'total': total,
            'verdict': verdict,
            'interpretation': interpretation
        }

    def _print_samples(self, category: str, samples: List[Dict], tokenize_fn):
        """Print sample sessions for inspection"""
        logger.info(f"\n📋 Sample {category} Sessions:")

        for i, session in enumerate(samples, 1):
            logger.info(f"\n  Session {i}:")
            logger.info(f"    User: {session.get('user_id')}")
            logger.info(f"    Time: {session.get('start_time')}")
            logger.info(f"    Events: {len(session.get('events', []))}")
            logger.info(f"    Malicious: {session.get('is_malicious', False)}")

            # Show tokens
            tokens = tokenize_fn(session)
            logger.info(f"    Sample tokens: {tokens[:3]}")

    def _print_results(self, results: Dict):
        """Print formatted results summary"""
        logger.info(f"\n" + "="*80)
        logger.info("SEMANTIC EVALUATION SUMMARY")
        logger.info("="*80)

        logger.info(f"\n📊 Sample Sizes:")
        logger.info(f"   Rare benign: {results['n_rare_benign']}")
        logger.info(f"   Similar attacks: {results['n_attacks']}")

        logger.info(f"\n📈 Surprise Scores:")
        logger.info(f"   Benign: μ={results['rare_benign_scores']['mean']:.2f}, "
                   f"σ={results['rare_benign_scores']['std']:.2f}")
        logger.info(f"   Attack: μ={results['attack_scores']['mean']:.2f}, "
                   f"σ={results['attack_scores']['std']:.2f}")

        logger.info(f"\n🎯 Key Metrics:")
        logger.info(f"   AUC: {results['classification']['auc']:.3f}")
        logger.info(f"   Benign Precision: {results['classification']['benign_precision']:.3f}")
        logger.info(f"   Attack Recall: {results['classification']['recall']:.3f}")
        logger.info(f"   Statistical Significance: {results['statistical_test']['is_significant']}")

        logger.info(f"\n🏆 Assessment: {results['assessment']['verdict']}")
        logger.info(f"   {results['assessment']['interpretation']}")
