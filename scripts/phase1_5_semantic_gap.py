"""
Phase 1.5: Semantic Gap Quantification and False Positive Analysis
CRITICAL - Must run before SAG to validate if semantic gap actually exists

IMPROVED: Now runs ACTUAL model training and prediction instead of estimating
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.data.lanl_loader import LANLLoader
from src.data.session_builder import SessionBuilder, SessionConfig
from src.evaluation.context_analysis import ContextWindowAnalyzer
from src.models.ngram_models import NgramLanguageModel, evaluate_ngram_model
from src.utils.reproducibility import set_seed
import gc
import psutil

# Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"phase1_5_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_memory_or_abort(operation_name: str, min_gb: float = 2.0):
    """Check memory before heavy operations and abort if insufficient"""
    try:
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < min_gb:
            logger.error(f"❌ Insufficient memory for {operation_name}")
            logger.error(f"   Available: {available_gb:.2f}GB < Required: {min_gb}GB")
            logger.error("   Solutions:")
            logger.error("   1. Close other applications")
            logger.error("   2. Reduce data size")
            logger.error("   3. Use smaller batch sizes")
            return False
        logger.info(f"✅ Memory check passed for {operation_name}: {available_gb:.2f}GB available")
        return True
    except Exception as e:
        logger.error(f"❌ Memory check failed: {e}")
        return False


def aggressive_cleanup(*objects):
    """Aggressively clean up memory"""
    for obj in objects:
        try:
            del obj
        except:
            pass
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    log_memory_usage("after_aggressive_cleanup")


def log_memory_usage(stage: str):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1024 / 1024 / 1024
        logger.info(f"💾 Memory at {stage}: {mem_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not log memory: {e}")


def categorize_false_positives(sessions, predictions, model_type="ngram"):
    """
    Categorize false positives into semantic categories

    Categories:
    1. Semantic Gap: Statistically rare + semantically benign (admin at 3AM)
    2. Data Quality: Missing features, parsing errors, etc.
    3. Model Limitation: Context not captured by model
    4. Possible Mislabel: Actually malicious but labeled benign
    """
    fp_categories = {
        'semantic_gap': [],
        'data_quality': [],
        'model_limitation': [],
        'possible_mislabel': []
    }

    for i, (session, pred) in enumerate(zip(sessions, predictions)):
        if session.get('is_malicious', False) == False and pred == 1:  # False positive
            category = categorize_single_fp(session, model_type)
            fp_categories[category].append({
                'session_id': i,
                'session': session,
                'reason': category
            })

    return fp_categories


def categorize_single_fp(session, model_type):
    """
    Categorize a single false positive session

    This is a heuristic-based categorization that should be refined
    with domain expertise and more sophisticated analysis.
    """
    # Extract session features for analysis
    events = session.get('events', [])
    if not events:
        return 'data_quality'

    # Check for admin activities at unusual times (semantic gap)
    is_admin_context = any(
        'admin' in str(event.get('user_id', '')).lower() or
        'system' in str(event.get('user_id', '')).lower()
        for event in events
    )

    # Check for unusual timing (outside business hours)
    session_start = session.get('start_time')
    if session_start:
        try:
            hour = pd.to_datetime(session_start).hour
            is_unusual_time = hour < 6 or hour > 22  # Before 6AM or after 10PM
        except:
            is_unusual_time = False
    else:
        is_unusual_time = False

    # Check for maintenance/service accounts
    is_maintenance = any(
        'backup' in str(event.get('auth_type', '')).lower() or
        'service' in str(event.get('user_id', '')).lower() or
        'maintenance' in str(event.get('auth_type', '')).lower()
        for event in events
    )

    # Semantic gap: Admin/maintenance at unusual times
    if (is_admin_context or is_maintenance) and is_unusual_time:
        return 'semantic_gap'

    # Data quality issues: Very short sessions, missing data
    if len(events) < 3:
        return 'data_quality'

    # Model limitation: Complex patterns not captured by n-gram
    # This is harder to detect - for now, assume other cases are model limitations
    return 'model_limitation'


def test_feature_informativeness(all_sessions):
    """
    Test if features capture semantics (CRITICAL from co-advisor feedback)
    """
    logger.info("🧪 Testing feature informativeness...")

    # Extract ONLY features (no sequences)
    all_labels = [0] * len([s for s in all_sessions if not s['is_malicious']]) + \
                 [1] * len([s for s in all_sessions if s['is_malicious']])

    X = []
    for session in all_sessions:
        features = get_session_features(session)
        # Convert to numeric
        feature_vector = [
            features.get('n_events', 0),
            features.get('duration', 0),
            1 if features.get('is_admin', False) else 0,
            features.get('hour', 0),
            1 if features.get('is_business_hours', False) else 0,
            1 if features.get('is_unusual_time', False) else 0,
            features.get('unique_users', 0),
            features.get('auth_type_diversity', 0),
            1 if features.get('has_critical_auth', False) else 0,
            1 if features.get('cross_host_activity', False) else 0
        ]
        X.append(feature_vector)

    X = np.array(X)
    y = np.array(all_labels)

    # Train simple tree on JUST features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"📊 Feature-only AUC: {auc:.3f}")

    # Decision logic from co-advisor
    if auc < 0.60:
        logger.error("❌ CRITICAL: Features are NOT informative!")
        logger.error("   Problem: Current features don't capture semantic context")
        logger.error("   Solutions:")
        logger.error("   1. Improve feature engineering (consider LLM assistance)")
        logger.error("   2. Add more domain-specific features")
        logger.error("   3. Use end-to-end learning instead of SAG")
        return False, auc
    elif auc < 0.70:
        logger.warning("⚠️ Features are weakly informative")
        logger.warning("   Consider improving features before SAG")
        return True, auc
    else:
        logger.info("✅ Features ARE informative")
        logger.info("   Good foundation for SAG approach")
        return True, auc


def estimate_sag_computational_cost(n_sessions, avg_seq_length):
    """
    Estimate if SAG is computationally feasible (from co-advisor feedback)
    """
    L = avg_seq_length
    T = 100  # Trees in LightGBM
    F = 50   # Number of features

    # TreeSHAP distillation cost
    operations_per_session = L * L * T * F
    total_operations = operations_per_session * n_sessions

    # Rough estimates (very approximate)
    ops_per_second = 1e9  # 1 billion ops/sec on modern CPU
    hours_needed = total_operations / ops_per_second / 3600

    # AWS cost estimate (rough)
    cost_per_hour = 3  # c5.4xlarge
    estimated_cost = hours_needed * cost_per_hour

    logger.info(f"Sequence length (L): {L:.1f}")
    logger.info(f"Number of sessions: {n_sessions:,}")
    logger.info(f"Total operations: {total_operations:,.0f}")
    logger.info(f"Estimated time: {hours_needed:.1f} hours")
    logger.info(f"Estimated AWS cost: ${estimated_cost:.2f}")

    if hours_needed > 48:
        logger.error("❌ COMPUTATIONALLY INFEASIBLE")
        logger.error("   Solutions:")
        logger.error("   1. Reduce L (max_events per session)")
        logger.error("   2. Use sparse attention (only key positions)")
        logger.error("   3. Approximate TreeSHAP")
        logger.error("   4. Try simpler methods first")
        return False
    elif hours_needed > 12:
        logger.warning("⚠️ EXPENSIVE - consider alternatives")
        return True
    else:
        logger.info("✅ Computationally reasonable")
        return True


def analyze_semantic_gap_with_tests(fp_categories, total_fps, total_benign, features_informative, feature_auc, computationally_feasible):
    """
    Analyze the semantic gap and provide recommendations

    Returns:
    - semantic_gap_ratio: % of FPs that are semantic gap
    - should_proceed_sag: Whether SAG is justified
    - recommendations: List of recommendations
    """
    semantic_fps = len(fp_categories['semantic_gap'])
    semantic_gap_ratio = semantic_fps / total_fps if total_fps > 0 else 0

    logger.info("\n📊 SEMANTIC GAP ANALYSIS:")
    logger.info(f"  Semantic Gap FPs: {semantic_fps}/{total_fps} ({semantic_gap_ratio*100:.1f}%)")
    logger.info(f"  Data Quality FPs: {len(fp_categories['data_quality'])}")
    logger.info(f"  Model Limitation FPs: {len(fp_categories['model_limitation'])}")
    logger.info(f"  Possible Mislabels: {len(fp_categories['possible_mislabel'])}")

    # Decision logic
    should_proceed_sag = semantic_gap_ratio > 0.5  # >50% of FPs are semantic
    overall_fp_rate = total_fps / total_benign if total_benign > 0 else 0

    recommendations = []

    if should_proceed_sag:
        recommendations.append("✅ SAG is justified - semantic gap is significant")
        recommendations.append(f"   {semantic_gap_ratio*100:.1f}% of false positives are semantically benign")
    else:
        recommendations.append("⚠️ SAG may not be justified")
        recommendations.append(f"   Only {semantic_gap_ratio*100:.1f}% of FPs are semantic gap")
        recommendations.append("   Consider simpler approaches first:")
        recommendations.append("   - Improve feature engineering")
        recommendations.append("   - Try context-aware n-grams")
        recommendations.append("   - Fix data quality issues")

    if overall_fp_rate < 0.01:  # <1% FP rate
        recommendations.append("✅ Overall FP rate is very low - current approach is good")
    elif overall_fp_rate > 0.10:  # >10% FP rate
        recommendations.append("❌ Overall FP rate is high - need significant improvement")

    if len(fp_categories['data_quality']) > 0.3 * total_fps:
        recommendations.append("⚠️ High data quality issues - fix data preprocessing first")

    return {
        'semantic_gap_ratio': semantic_gap_ratio,
        'should_proceed_sag': should_proceed_sag,
        'overall_fp_rate': overall_fp_rate,
        'recommendations': recommendations,
        'fp_breakdown': {k: len(v) for k, v in fp_categories.items()}
    }


def run_actual_semantic_gap_analysis():
    """
    Run ACTUAL semantic gap analysis with real model training and prediction
    """
    logger.info("🚀 PHASE 1.5: ACTUAL SEMANTIC GAP ANALYSIS")
    logger.info("="*80)

    # Check memory before starting
    if not check_memory_or_abort("semantic_gap_analysis", min_gb=3.0):
        return 1

    # Load data (conservative for analysis)
    logger.info("📂 Loading data for analysis...")
    loader = LANLLoader(Path("data/raw/lanl"))

    # Quick check for attack days
    redteam_file = Path("data/raw/lanl/redteam.txt")
    if not redteam_file.exists():
        logger.error("❌ No red team labels!")
        return 1

    redteam_quick = pd.read_csv(redteam_file, header=None,
                               names=['time', 'user', 'src_computer', 'dst_computer'])
    redteam_quick['day'] = (redteam_quick['time'] / 86400).astype(int) + 1
    attack_days = sorted(redteam_quick['day'].unique())
    logger.info(f"📅 Attack days: {attack_days}")

    # Load minimal data for analysis (1-2 attack days + buffer)
    days_to_load = list(range(
        attack_days[0] - 1,
        min(attack_days[0] + 3, attack_days[0] + 5)  # Max 5 days for analysis
    ))
    logger.info(f"📊 Loading {len(days_to_load)} days for analysis")

    auth_df, redteam_df = loader.load_sample(days=days_to_load)

    # Normalize column names
    if 'user' in redteam_df.columns and 'user_id' not in redteam_df.columns:
        redteam_df['user_id'] = redteam_df['user']

    # Clean up redteam_quick
    aggressive_cleanup(redteam_quick)
    log_memory_usage("After data load")

    # Limit for analysis
    MAX_EVENTS = 500_000  # Conservative for analysis
    if len(auth_df) > MAX_EVENTS:
        logger.warning(f"⚠️ Downsampling for analysis: {len(auth_df):,} → {MAX_EVENTS:,}")
        auth_df = auth_df.sample(n=MAX_EVENTS, random_state=42)

    # Build sessions
    logger.info("🔧 Building sessions for analysis...")
    config = SessionConfig(
        timeout_minutes=30,
        min_events=3,
        max_events=100,
        labeling="window",
        label_window_minutes=240
    )
    builder = SessionBuilder(config)
    all_sessions = builder.build_sessions(auth_df, redteam_df, train_mode=False)

    # Clean up dataframes
    aggressive_cleanup(auth_df, redteam_df)
    log_memory_usage("After session building")

    # Filter sessions
    benign = [s for s in all_sessions if not s['is_malicious']]
    malicious = [s for s in all_sessions if s['is_malicious']]

    logger.info(f"📊 Dataset for analysis:")
    logger.info(f"  Benign: {len(benign)}")
    logger.info(f"  Malicious: {len(malicious)}")

    if len(malicious) < 10:
        logger.error("❌ Need >= 10 malicious samples for analysis")
        return 1

    # Sample benign for analysis (keep all malicious)
    MAX_BENIGN_ANALYSIS = 5000
    if len(benign) > MAX_BENIGN_ANALYSIS:
        logger.info(f"📊 Sampling {MAX_BENIGN_ANALYSIS} benign for analysis")
        import random
        random.seed(42)
        benign = random.sample(benign, MAX_BENIGN_ANALYSIS)

    log_memory_usage("After session filtering")

    # SPLIT DATA FOR PROPER EVALUATION
    logger.info("🔀 Splitting data for proper evaluation...")
    benign_train, benign_test = train_test_split(
        benign, test_size=0.3, random_state=42, stratify=[0] * len(benign)
    )

    logger.info(f"📊 Split: {len(benign_train)} train, {len(benign_test)} test benign")
    logger.info(f"📊 Malicious for testing: {len(malicious)}")

    # PREPARE SEQUENCE DATA FOR N-GRAM MODEL
    logger.info("🔧 Preparing sequence data for n-gram model...")

    def sessions_to_sequences(sessions):
        """Convert sessions to sequences for n-gram modeling"""
        sequences = []
        for session in sessions:
            events = session.get('events', [])
            # Create simple event representation for n-gram
            event_sequence = [
                f"{e.get('auth_type', 'UNK')}_{e.get('outcome', 'UNK')}"
                for e in events
            ]
            if len(event_sequence) >= 2:  # Need at least 2 events for n-gram
                sequences.append(event_sequence)
        return sequences

    train_sequences = sessions_to_sequences(benign_train)
    test_benign_sequences = sessions_to_sequences(benign_test)
    test_malicious_sequences = sessions_to_sequences(malicious)

    logger.info(f"📊 Training sequences: {len(train_sequences)}")
    logger.info(f"📊 Test benign sequences: {len(test_benign_sequences)}")
    logger.info(f"📊 Test malicious sequences: {len(test_malicious_sequences)}")

    if len(train_sequences) == 0 or len(test_benign_sequences) == 0:
        logger.error("❌ No valid sequences for analysis")
        return 1

    # TRAIN N-GRAM MODEL
    logger.info("🧠 Training n-gram model...")
    if not check_memory_or_abort("ngram_training", min_gb=2.0):
        return 1

    try:
        ngram_model = NgramLanguageModel(n=5, smoothing='laplace')
        ngram_model.fit(train_sequences)
        logger.info("✅ N-gram model trained successfully")
    except Exception as e:
        logger.error(f"❌ Failed to train n-gram model: {e}")
        return 1

    log_memory_usage("After ngram training")

    # EVALUATE MODEL AND GET ACTUAL PREDICTIONS
    logger.info("🎯 Evaluating model and getting predictions...")

    # Get anomaly scores for test data
    try:
        # Calculate max surprise scores for each test sequence
        benign_scores = []
        for seq in test_benign_sequences:
            if len(seq) > 0:
                surprise_scores = ngram_model.surprise_scores(seq)
                benign_scores.append(surprise_scores.max() if len(surprise_scores) > 0 else 0.0)
            else:
                benign_scores.append(0.0)

        malicious_scores = []
        for seq in test_malicious_sequences:
            if len(seq) > 0:
                surprise_scores = ngram_model.surprise_scores(seq)
                malicious_scores.append(surprise_scores.max() if len(surprise_scores) > 0 else 0.0)
            else:
                malicious_scores.append(0.0)

        logger.info(f"📊 Got scores for {len(benign_scores)} benign, {len(malicious_scores)} malicious")

    except Exception as e:
        logger.error(f"❌ Failed to get predictions: {e}")
        return 1

    # SET THRESHOLD AND FIND ACTUAL FALSE POSITIVES
    logger.info("⚖️ Setting threshold and identifying false positives...")

    # Use 95th percentile of benign scores as threshold
    if len(benign_scores) > 0:
        threshold = np.percentile(benign_scores, 95)
        logger.info(f"📊 Threshold (95th percentile): {threshold:.4f}")
    else:
        logger.error("❌ No benign scores available")
        return 1

    # Get predictions
    benign_predictions = [1 if score > threshold else 0 for score in benign_scores]
    malicious_predictions = [1 if score > threshold else 0 for score in malicious_scores]

    # Find actual false positives
    actual_fps = [
        (session, score) for session, score, pred in
        zip(benign_test, benign_scores, benign_predictions)
        if pred == 1  # Model predicted malicious, but actually benign
    ]

    logger.info(f"📊 Found {len(actual_fps)} ACTUAL false positives")
    logger.info(f"📊 Actual FP rate: {len(actual_fps) / len(benign_test) * 100:.2f}%")

    # CATEGORIZE ACTUAL FALSE POSITIVES
    logger.info("🔍 Categorizing actual false positives...")

    fp_categories = {
        'semantic_gap': [],
        'data_quality': [],
        'model_limitation': [],
        'possible_mislabel': []
    }

    for session, score in actual_fps:
        category, reason = categorize_single_fp(session, "ngram")
        fp_categories[category].append({
            'session': session,
            'score': score,
            'reason': reason,
            'features': get_session_features(session)
        })

    # ANALYZE SEMANTIC GAP
    total_fps = len(actual_fps)
    semantic_fps = len(fp_categories['semantic_gap'])
    semantic_gap_ratio = semantic_fps / total_fps if total_fps > 0 else 0

    logger.info("\n" + "="*80)
    logger.info("📊 SEMANTIC GAP ANALYSIS (ACTUAL MEASUREMENTS)")
    logger.info("="*80)
    logger.info(f"Total actual FPs: {total_fps}")
    logger.info(f"Semantic gap FPs: {semantic_fps} ({semantic_gap_ratio*100:.1f}%)")
    logger.info(f"Data quality FPs: {len(fp_categories['data_quality'])}")
    logger.info(f"Model limitation FPs: {len(fp_categories['model_limitation'])}")
    logger.info(f"Possible mislabel FPs: {len(fp_categories['possible_mislabel'])}")

    # TEST 1: FEATURE QUALITY ASSESSMENT (CRITICAL)
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING FEATURE INFORMATIVENESS")
    logger.info("="*80)

    features_informative, feature_auc = test_feature_informativeness(benign + malicious)

    # TEST 2: COMPUTATIONAL COST ESTIMATE
    logger.info("\n" + "="*80)
    logger.info("💰 COMPUTATIONAL COST ESTIMATE FOR SAG")
    logger.info("="*80)

    # Estimate average sequence length
    all_sequences = train_sequences + test_benign_sequences + test_malicious_sequences
    avg_seq_length = sum(len(seq) for seq in all_sequences) / len(all_sequences) if all_sequences else 50

    n_sessions = len(benign) + len(malicious)
    computationally_feasible = estimate_sag_computational_cost(n_sessions, avg_seq_length)

    # DECISION LOGIC WITH ADDITIONAL TESTS
    analysis = analyze_semantic_gap_with_tests(fp_categories, total_fps, len(benign_test),
                                             features_informative, feature_auc,
                                             computationally_feasible)

    # SAVE RESULTS
    output_dir = Path("experiments/phase1_5")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "actual_semantic_gap_analysis.pkl", "wb") as f:
        pickle.dump({
            'analysis': analysis,
            'fp_categories': fp_categories,
            'actual_fps': actual_fps,
            'model_results': {
                'benign_scores': benign_scores,
                'malicious_scores': malicious_scores,
                'threshold': threshold,
                'benign_predictions': benign_predictions,
                'malicious_predictions': malicious_predictions
            },
            'n_benign_train': len(benign_train),
            'n_benign_test': len(benign_test),
            'n_malicious': len(malicious),
            'timestamp': datetime.now().isoformat()
        }, f)

    logger.info(f"✅ Saved actual analysis to {output_dir}")

    # PRINT RECOMMENDATIONS
    logger.info("\n" + "="*80)
    logger.info("📋 RECOMMENDATIONS:")
    logger.info("="*80)
    for rec in analysis['recommendations']:
        logger.info(f"  {rec}")

    logger.info("\n" + "="*80)
    if analysis['should_proceed_sag']:
        logger.info("✅ PROCEED TO SAG - Semantic gap justifies complex approach")
        logger.info(f"   Justification: {semantic_gap_ratio*100:.1f}% of FPs are semantic gap")
    else:
        logger.info("⚠️ RECONSIDER - Semantic gap may not justify SAG complexity")
        logger.info(f"   Only {semantic_gap_ratio*100:.1f}% of FPs are semantic gap")

    # Clean up
    aggressive_cleanup(benign, malicious, all_sessions, ngram_model)

    return 0 if analysis['should_proceed_sag'] else 1


def get_session_features(session):
    """Extract features from session for semantic analysis"""
    features = {}

    # Basic session info
    features['n_events'] = len(session.get('events', []))
    features['duration'] = session.get('duration_minutes', 0)

    # User info
    events = session.get('events', [])
    if events:
        user_ids = [str(e.get('user_id', '')) for e in events]
        features['unique_users'] = len(set(user_ids))
        features['is_admin'] = any('admin' in uid.lower() or 'system' in uid.lower() for uid in user_ids)

        # Time features
        start_time = session.get('start_time')
        if start_time:
            try:
                dt = pd.to_datetime(start_time)
                features['hour'] = dt.hour
                features['day_of_week'] = dt.dayofweek
                features['is_business_hours'] = 9 <= dt.hour <= 17 and dt.dayofweek < 5
                features['is_unusual_time'] = dt.hour < 6 or dt.hour > 22
            except:
                features['hour'] = -1
                features['is_business_hours'] = False
                features['is_unusual_time'] = False

        # Auth types
        auth_types = [str(e.get('auth_type', '')) for e in events]
        features['auth_type_diversity'] = len(set(auth_types))
        features['has_critical_auth'] = any('Kerberos' in at or 'NTLM' in at for at in auth_types)

        # Host patterns
        src_hosts = [str(e.get('src_comp_id', '')) for e in events]
        dst_hosts = [str(e.get('dst_comp_id', '')) for e in events]
        features['unique_src_hosts'] = len(set(src_hosts))
        features['unique_dst_hosts'] = len(set(dst_hosts))
        features['cross_host_activity'] = len(set(src_hosts + dst_hosts)) > 1

    return features


if __name__ == "__main__":
    sys.exit(run_actual_semantic_gap_analysis())
