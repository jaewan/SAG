"""
Phase 1.5: Detailed False Positive Analysis
Run models on benign sessions to get actual FP predictions and categorize them

IMPROVED: Now uses ACTUAL model training and prediction instead of estimates
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.data.lanl_loader import LANLLoader
from src.data.session_builder import SessionBuilder, SessionConfig
from src.evaluation.context_analysis import ContextWindowAnalyzer
from src.models.ngram_models import NgramLanguageModel
from src.features.semantic_features import extract_semantic_features_batch, SemanticFeatureExtractor
from src.utils.reproducibility import set_seed
import gc
import psutil
import random

# Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"phase1_5_detailed_{datetime.now():%Y%m%d_%H%M%S}.log"),
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


def categorize_fp_with_semantic_features(event, model_type="semantic"):
    """
    Categorize false positive using semantic features from correlated events
    """
    # Extract semantic features for this event
    features_df = extract_semantic_features_batch([event])
    if len(features_df) == 0:
        return 'data_quality', "No semantic features available"

    features = features_df.iloc[0].to_dict()

    # Category 1: Semantic Gap (admin/maintenance at unusual times)
    if (features.get('user_is_admin', False) and not features.get('is_business_hours', True)):
        return 'semantic_gap', "Admin activity outside business hours"

    if not features.get('is_business_hours', True) and features.get('is_maintenance_window', False):
        return 'semantic_gap', "Activity during maintenance window"

    # Category 2: Process Chain Issues
    if features.get('parent_child_suspicious', 0) > 0.5:
        return 'semantic_gap', f"Suspicious process chain (score: {features.get('parent_child_suspicious', 0):.2f})"

    if not features.get('user_process_legitimate', True):
        return 'semantic_gap', "User not authorized for these processes"

    # Category 3: Network Context Issues
    if features.get('network_direction') == 'external':
        return 'semantic_gap', "External network connections"

    if features.get('unusual_network_activity', False):
        return 'semantic_gap', "Unusual network activity patterns"

    # Category 4: Data Quality Issues
    if len(event.get('related_processes', [])) == 0:
        return 'data_quality', "No process context available"

    if len(event.get('auth_event', {})) == 0:
        return 'data_quality', "No authentication context"

    # Category 5: Model Limitations (complex patterns)
    if features.get('has_suspicious_processes', False):
        return 'model_limitation', "Suspicious processes detected"

    # Category 6: Possible Mislabels (suspicious patterns)
    if (not features.get('is_business_hours', True) and
        features.get('user_is_admin', False) == False and
        features.get('parent_child_suspicious', 0) > 0.3):
        return 'possible_mislabel', "Non-admin suspicious activity at unusual time"

    # Default to model limitation
    return 'model_limitation', "Pattern not captured by semantic model"


def test_simple_baselines(all_events):
    """
    Test simple baselines using semantic features (no sequences)
    This helps determine if SAG complexity is justified
    """
    logger.info("🧪 Testing simple baselines with semantic features only...")

    # Extract semantic features for all events
    features_df = extract_semantic_features_batch(all_events)

    if len(features_df) == 0:
        logger.error("❌ No semantic features extracted")
        return {}

    # Prepare feature matrix
    feature_cols = ['user_is_admin', 'parent_child_suspicious', 'is_business_hours',
                   'is_maintenance_window', 'has_suspicious_processes']

    # Fill missing columns with defaults
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].values
    y = features_df['is_malicious'].astype(int).values

    if len(X) == 0 or len(np.unique(y)) < 2:
        logger.error("❌ Not enough data for baseline testing")
        return {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    results = {}

    # Baseline 1: LightGBM (tree-based)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_test, lgb_pred_proba)
        results['lightgbm'] = {
            'auc': lgb_auc,
            'model': lgb_model
        }
        logger.info(f"   LightGBM AUC: {lgb_auc:.3f}")
    except ImportError:
        logger.warning("   ⚠️ LightGBM not available - skipping")
        results['lightgbm'] = {'auc': 0.5}

    # Baseline 2: Random Forest
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        results['random_forest'] = {
            'auc': rf_auc,
            'model': rf_model
        }
        logger.info(f"   Random Forest AUC: {rf_auc:.3f}")
    except:
        logger.warning("   ⚠️ Random Forest failed - skipping")
        results['random_forest'] = {'auc': 0.5}

    # Baseline 3: Logistic Regression
    try:
        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred_proba)
        results['logistic_regression'] = {
            'auc': lr_auc,
            'model': lr_model
        }
        logger.info(f"   Logistic Regression AUC: {lr_auc:.3f}")
    except:
        logger.warning("   ⚠️ Logistic Regression failed - skipping")
        results['logistic_regression'] = {'auc': 0.5}

    # Baseline 4: Rule-based (semantic heuristics)
    rule_predictions = []
    for features in X_test:
        # Semantic rule: flag if suspicious process chain + not business hours + not admin
        parent_child_suspicious = features[1]  # parent_child_suspicious
        is_business_hours = features[2]       # is_business_hours
        user_is_admin = features[0]           # user_is_admin

        rule_score = 0.0
        if parent_child_suspicious > 0.5 and not is_business_hours and not user_is_admin:
            rule_score = 0.8  # High suspicion
        elif parent_child_suspicious > 0.3 and not is_business_hours:
            rule_score = 0.4  # Medium suspicion

        rule_predictions.append(rule_score)

    rule_auc = roc_auc_score(y_test, rule_predictions)
    results['rule_based'] = {
        'auc': rule_auc,
        'predictions': rule_predictions
    }
    logger.info(f"   Rule-based AUC: {rule_auc:.3f}")

    # Analyze results
    baseline_aucs = [r.get('auc', 0.5) for r in results.values() if 'auc' in r]
    best_baseline_auc = max(baseline_aucs) if baseline_aucs else 0.5

    logger.info(f"📊 Best baseline AUC: {best_baseline_auc:.3f}")

    if best_baseline_auc > 0.85:
        logger.warning("⚠️ Simple baselines achieve high performance")
        logger.warning("   SAG may not be necessary - consider simpler approaches")
    elif best_baseline_auc > 0.75:
        logger.info("✅ Baselines show moderate performance")
        logger.info("   SAG might provide incremental improvement")
    else:
        logger.info("✅ Baselines show poor performance")
        logger.info("   SAG could provide significant improvement")

    return results


def test_feature_informativeness(all_events):
    """
    Test if semantic features capture semantics (CRITICAL from co-advisor feedback)
    """
    logger.info("🧪 Testing semantic feature informativeness...")

    # Extract features for all events
    features_df = extract_semantic_features_batch(all_events)

    if len(features_df) == 0:
        logger.error("❌ No features extracted")
        return False, 0.5

    # Prepare feature matrix
    feature_cols = ['user_is_admin', 'parent_child_suspicious', 'is_business_hours',
                   'is_maintenance_window', 'has_suspicious_processes']

    # Fill missing columns with defaults
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].values
    y = features_df['is_malicious'].astype(int).values

    if len(X) == 0 or len(np.unique(y)) < 2:
        logger.error("❌ Not enough data for feature testing")
        return False, 0.5

    # Train simple tree on semantic features only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"📊 Semantic feature-only AUC: {auc:.3f}")

    # Decision logic from co-advisor (adapted for semantic features)
    if auc < 0.60:
        logger.error("❌ CRITICAL: Semantic features are NOT informative!")
        logger.error("   Problem: Current semantic features don't capture malicious context")
        logger.error("   Solutions:")
        logger.error("   1. Improve semantic feature engineering")
        logger.error("   2. Add more domain-specific semantic features")
        logger.error("   3. Check if correlated data is properly loaded")
        return False, auc
    elif auc < 0.70:
        logger.warning("⚠️ Semantic features are weakly informative")
        logger.warning("   Consider improving semantic features before SAG")
        return True, auc
    else:
        logger.info("✅ Semantic features ARE informative")
        logger.info("   Good foundation for SAG approach")
        return True, auc


def test_memory_scaling_progressive(loader, redteam_file, max_test_size_gb=8.0):
    """
    Test memory usage at different data scales progressively
    Returns the maximum safe data size for the current system
    """
    logger.info("🔬 PROGRESSIVE MEMORY SCALING TEST")
    logger.info("="*80)

    # Test sizes in events (not sessions)
    test_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    # Load attack days for consistent testing
    redteam_quick = pd.read_csv(redteam_file, header=None,
                               names=['time', 'user', 'src_computer', 'dst_computer'])
    redteam_quick['day'] = (redteam_quick['time'] / 86400).astype(int) + 1
    attack_days = sorted(redteam_quick['day'].unique())

    results = []
    safe_size = None

    for test_size in test_sizes:
        logger.info(f"\n🧪 Testing with {test_size:,}","events...")

        try:
            # Load small sample
            days_to_load = list(range(
                attack_days[0] - 1,
                min(attack_days[0] + 2, attack_days[0] + 3)  # Small window for testing
            ))

            auth_df, redteam_df = loader.load_sample(days=days_to_load)

            # Downsample to test size
            if len(auth_df) > test_size:
                auth_df = auth_df.sample(n=test_size, random_state=42)

            # Memory before
            mem_before = psutil.virtual_memory().used / 1e9
            logger.info(f"   Memory before: {mem_before:.2f}GB")

            # Build sessions (this is where memory explodes)
            config = SessionConfig(timeout_minutes=30, min_events=3, max_events=100)
            builder = SessionBuilder(config)

            # Test with streaming builder if available
            if hasattr(builder, 'build_sessions_streaming'):
                sessions = list(builder.build_sessions_streaming(
                    auth_df, redteam_df, train_mode=False, chunk_size=5000
                ))
            else:
                sessions = builder.build_sessions(auth_df, redteam_df, train_mode=False)

            # Memory after
            mem_after = psutil.virtual_memory().used / 1e9
            mem_used = mem_after - mem_before

            # Calculate efficiency metrics
            n_sessions = len(sessions)
            avg_events_per_session = len(auth_df) / n_sessions if n_sessions > 0 else 0

            logger.info(f"   Memory after: {mem_after:.2f}GB")
            logger.info(f"   Memory used: {mem_used:.2f}GB")
            logger.info(f"   Sessions created: {n_sessions:,}")
            logger.info(f"   Avg events/session: {avg_events_per_session:.1f}")
            logger.info(f"   Memory per event: {mem_used / len(auth_df) * 1e6:.2f}KB")
            logger.info(f"   Memory per session: {mem_used / n_sessions * 1e6:.2f}KB")

            results.append({
                'test_size': test_size,
                'mem_used_gb': mem_used,
                'n_sessions': n_sessions,
                'efficiency': mem_used / len(auth_df) if len(auth_df) > 0 else 0
            })

            # Check if this size is safe
            available_gb = psutil.virtual_memory().available / 1e9
            if mem_used > available_gb * 0.8:  # Use >80% of available memory
                logger.warning(f"   ⚠️ High memory usage: {mem_used:.2f}GB used")
                if safe_size is None:
                    safe_size = test_sizes[test_sizes.index(test_size) - 1] if test_sizes.index(test_size) > 0 else test_size // 10
                break
            else:
                safe_size = test_size
                logger.info("   ✅ Safe for this scale")

            # Clean up
            del auth_df, redteam_df, sessions
            gc.collect()

        except Exception as e:
            logger.error(f"   ❌ Failed at {test_size:,} events: {e}")
            if safe_size is None:
                safe_size = test_sizes[test_sizes.index(test_size) - 1] if test_sizes.index(test_size) > 0 else test_size // 10
            break

    # Analyze results
    logger.info("\n📊 PROGRESSIVE TEST RESULTS:")
    logger.info(f"Safe size: {safe_size:,}","events")

    # Extrapolate to full dataset
    full_size = 60_000_000  # LANL full size
    if safe_size and safe_size > 0:
        scale_factor = full_size / safe_size
        estimated_full_mem = results[-1]['mem_used_gb'] * scale_factor if results else 0
        logger.info(f"Estimated full dataset memory: {estimated_full_mem:.1f}GB")

        available_gb = psutil.virtual_memory().total / 1e9
        if estimated_full_mem > available_gb * 0.9:
            logger.error(f"❌ Full dataset likely to OOM: {estimated_full_mem:.1f}GB > {available_gb * 0.9:.1f}GB")
            logger.error("   Solutions:")
            logger.error("   1. Use streaming session builder")
            logger.error("   2. Reduce max_events per session")
            logger.error("   3. Process in smaller batches")
            logger.error("   4. Use efficient session storage")
        else:
            logger.info("✅ Full dataset should fit in memory")

    return safe_size, results


def estimate_sag_computational_cost(n_sessions, avg_seq_length):
    """
    Estimate if SAG is computationally feasible (from co-advisor feedback)
    """
    L = avg_seq_length
    T = 100  # Trees in LightGBM
    F = 50   # Number of features
    N = 3    # ✅ FIX: Changed from n=5 to n=3 for n-gram model

    # TreeSHAP distillation cost
    operations_per_session = L * L * T * F * N
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


def run_detailed_fp_analysis():
    """
    Run detailed false positive analysis with ACTUAL model predictions
    """
    logger.info("🚀 PHASE 1.5: DETAILED FALSE POSITIVE ANALYSIS")
    logger.info("="*80)

    # Check memory before starting
    if not check_memory_or_abort("detailed_fp_analysis", min_gb=3.0):
        return 1

    # Load data using new multi-source loader
    logger.info("📂 Loading data for detailed analysis with multi-source correlation...")
    loader = LANLLoader(Path("data/raw/lanl"))

    # Check if all required files exist
    required_files = ['auth.txt', 'proc.txt', 'flows.txt', 'dns.txt', 'redteam.txt']
    missing_files = [f for f in required_files if not (Path("data/raw/lanl") / f).exists()]
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return 1

    # STEP 1: PROGRESSIVE MEMORY TESTING (NEW)
    logger.info("🔬 STEP 1: PROGRESSIVE MEMORY TESTING")
    logger.info("="*80)

    # Test memory scaling with new multi-source approach
    safe_size, test_results = test_memory_scaling_progressive(loader, Path("data/raw/lanl/redteam.txt"))

    if safe_size is None or safe_size < 10000:
        logger.error("❌ Cannot determine safe data size - aborting")
        return 1

    logger.info(f"✅ Safe size determined: {safe_size:,}","events")

    # Adjust data size based on progressive testing
    target_size = min(safe_size, 100_000)  # Cap at 100K for analysis with multiple sources
    logger.info(f"📊 Using {target_size:,}","events for analysis")

    # Load attack days for targeting
    redteam_df = loader._load_redteam_data()
    if len(redteam_df) == 0:
        logger.error("❌ No red team data found")
        return 1

    attack_days = sorted(redteam_df['day'].unique())
    logger.info(f"📅 Attack days: {attack_days}")

    # Load attack days + small buffer (LANL dataset starts on day 91)
    days_to_load = list(range(
        max(91, attack_days[0] - 1),  # Don't go before day 91
        min(attack_days[0] + 3, attack_days[0] + 5)  # Max 5 days for analysis
    ))
    logger.info(f"📊 Loading {len(days_to_load)} days")

    # Load correlated events using new multi-source approach
    logger.info("🔗 Loading and correlating multiple data sources...")
    correlated_events, quality_report = loader.load_full_context(
        days=days_to_load,
        max_rows=target_size,
        correlation_window_sec=300  # 5 minute correlation window
    )

    # Check correlation quality (CRITICAL from peer review)
    logger.info(f"📊 Correlation quality: {quality_report['correlation_rate']*100:.1f}% events have context")
    if quality_report['status'] == 'poor':
        logger.error("❌ CORRELATION QUALITY IS POOR - This is a critical blocker!")
        logger.error("   Fix correlation before proceeding with SAG")
        return 1

    log_memory_usage("After multi-source data loading")

    # Filter to events with related processes (for semantic analysis)
    logger.info("🔍 Filtering to events with semantic context...")
    events_with_processes = [
        event for event in correlated_events
        if len(event.get('related_processes', [])) > 0
    ]

    logger.info(f"📊 Events with process context: {len(events_with_processes)}/{len(correlated_events)}")

    if len(events_with_processes) == 0:
        logger.error("❌ No events with process context for semantic analysis")
        return 1

    # Split into benign and malicious
    benign_events = [e for e in events_with_processes if not e['is_malicious']]
    malicious_events = [e for e in events_with_processes if e['is_malicious']]

    logger.info(f"📊 Dataset:")
    logger.info(f"  Benign: {len(benign_events)}")
    logger.info(f"  Malicious: {len(malicious_events)}")

    if len(malicious_events) < 10:
        logger.error("❌ Need >= 10 malicious samples")
        return 1

    # Sample benign for analysis
    MAX_BENIGN_ANALYSIS = 2000
    if len(benign_events) > MAX_BENIGN_ANALYSIS:
        logger.info(f"📊 Sampling {MAX_BENIGN_ANALYSIS} benign for detailed analysis")
        random.seed(42)
        benign_events = random.sample(benign_events, MAX_BENIGN_ANALYSIS)

    log_memory_usage("After benign sampling")

    # SPLIT DATA FOR PROPER EVALUATION
    logger.info("🔀 Splitting data for proper evaluation...")
    benign_train, benign_test = train_test_split(
        benign_events, test_size=0.3, random_state=42
    )

    logger.info(f"📊 Split: {len(benign_train)} train, {len(benign_test)} test benign")
    logger.info(f"📊 Malicious for testing: {len(malicious_events)}")

    # PREPARE SEQUENCE DATA
    logger.info("🔧 Preparing sequence data...")

    def events_to_sequences(events):
        """Convert correlated events to sequences for n-gram modeling"""
        sequences = []
        for event in events:
            # Create sequence from auth events
            auth_event = event.get('auth_event', {})
            event_sequence = [f"{auth_event.get('auth_type', 'UNK')}_{auth_event.get('outcome', 'UNK')}"]

            # Add process events if available
            processes = event.get('related_processes', [])
            for proc in processes:
                event_sequence.append(f"PROC_{proc.get('process_name', 'UNK')}")

            if len(event_sequence) >= 2:  # Need at least 2 events for n-gram
                sequences.append(event_sequence)
        return sequences

    train_sequences = events_to_sequences(benign_train)
    test_benign_sequences = events_to_sequences(benign_test)
    test_malicious_sequences = events_to_sequences(malicious_events)

    logger.info(f"📊 Training sequences: {len(train_sequences)}")
    logger.info(f"📊 Test benign sequences: {len(test_benign_sequences)}")
    logger.info(f"📊 Test malicious sequences: {len(test_malicious_sequences)}")

    if len(train_sequences) == 0 or len(test_benign_sequences) == 0:
        logger.error("❌ No valid sequences for analysis")
        return 1

    # TRAIN N-GRAM MODEL (with vocabulary pruning)
    logger.info("🧠 Training n-gram model with vocabulary pruning...")
    if not check_memory_or_abort("ngram_training", min_gb=2.0):
        return 1

    try:
        # ✅ NEW: Use pruned n-gram model to prevent memory explosion
        from src.models.ngram_models import PrunedNgramLanguageModel
        ngram_model = PrunedNgramLanguageModel(
            n=3,
            smoothing='laplace',
            max_vocab_size=50000,  # Limit vocabulary size
            min_count=2            # Minimum frequency for tokens
        )
        ngram_model.fit(train_sequences)
        logger.info("✅ Pruned n-gram model trained successfully")
        logger.info(f"   Vocabulary size: {len(ngram_model.pruned_vocab) if ngram_model.pruned_vocab else 'N/A'}")
    except Exception as e:
        logger.error(f"❌ Failed to train pruned n-gram model: {e}")
        logger.info("   Falling back to regular n-gram model...")
        try:
            ngram_model = NgramLanguageModel(n=3, smoothing='laplace')
            ngram_model.fit(train_sequences)
            logger.info("✅ Regular n-gram model trained successfully")
        except Exception as e2:
            logger.error(f"❌ Failed to train n-gram model: {e2}")
            return 1

    log_memory_usage("After ngram training")

    # EXTRACT SEMANTIC FEATURES FOR ALL EVENTS
    logger.info("🎯 Extracting semantic features for all events...")

    # Extract features for training data
    train_features_df = extract_semantic_features_batch(benign_train)
    test_benign_features_df = extract_semantic_features_batch(benign_test)
    test_malicious_features_df = extract_semantic_features_batch(malicious_events)

    # Validate feature quality (CRITICAL from peer review)
    logger.info("🧪 Validating feature quality...")
    total_features = len(train_features_df) + len(test_benign_features_df) + len(test_malicious_features_df)

    # Check if features are informative (not all defaults)
    from src.features.semantic_features import has_meaningful_features

    meaningful_count = 0
    for df in [train_features_df, test_benign_features_df, test_malicious_features_df]:
        for _, row in df.iterrows():
            if has_meaningful_features(row.to_dict()):
                meaningful_count += 1

    meaningful_rate = meaningful_count / total_features if total_features > 0 else 0
    logger.info(f"📊 Feature informativeness: {meaningful_rate*100:.1f}% meaningful features")

    if meaningful_rate < 0.1:  # Less than 10% meaningful features
        logger.error("❌ CRITICAL: Features are NOT informative!")
        logger.error("   Problem: Current semantic features don't capture malicious context")
        logger.error("   Solutions:")
        logger.error("   1. Improve semantic feature engineering")
        logger.error("   2. Add more domain-specific semantic features")
        logger.error("   3. Check if correlated data is properly loaded")
        return 1

    logger.info(f"📊 Extracted features - Train: {len(train_features_df)}, Test benign: {len(test_benign_features_df)}, Test malicious: {len(test_malicious_features_df)}")

    # GET ACTUAL PREDICTIONS USING SEMANTIC FEATURES
    logger.info("🎯 Getting actual predictions using semantic features...")

    # Use semantic features to train a simple model for comparison
    try:
        # Prepare feature matrices
        feature_cols = ['user_is_admin', 'parent_child_suspicious', 'is_business_hours',
                       'is_maintenance_window', 'has_suspicious_processes']

        # Fill missing features with defaults
        for df in [train_features_df, test_benign_features_df, test_malicious_features_df]:
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

        X_train = train_features_df[feature_cols].values
        y_train = train_features_df['is_malicious'].astype(int).values

        X_test_benign = test_benign_features_df[feature_cols].values
        X_test_malicious = test_malicious_features_df[feature_cols].values

        # Train simple classifier on semantic features
        from sklearn.ensemble import RandomForestClassifier
        feature_model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_model.fit(X_train, y_train)

        # Get predictions
        benign_scores = feature_model.predict_proba(X_test_benign)[:, 1]
        malicious_scores = feature_model.predict_proba(X_test_malicious)[:, 1]

        logger.info(f"📊 Got semantic feature scores for {len(benign_scores)} benign, {len(malicious_scores)} malicious")

    except Exception as e:
        logger.error(f"❌ Failed to train feature-based model: {e}")
        logger.info("Falling back to n-gram model for predictions...")
        # Fallback to n-gram if semantic features fail

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

    # Find actual false positives (events predicted as malicious but actually benign)
    actual_fps = [
        (event, score) for event, score, pred in
        zip(benign_test, benign_scores, benign_predictions)
        if pred == 1  # Model predicted malicious, but actually benign
    ]

    logger.info(f"📊 Found {len(actual_fps)} ACTUAL false positives")
    logger.info(f"📊 Actual FP rate: {len(actual_fps) / len(benign_test) * 100:.2f}%")

    # CATEGORIZE ACTUAL FALSE POSITIVES USING SEMANTIC FEATURES
    logger.info("🔍 Categorizing actual false positives using semantic features...")

    fp_categories = {
        'semantic_gap': [],
        'data_quality': [],
        'model_limitation': [],
        'possible_mislabel': []
    }

    for event, score in actual_fps:
        category, reason = categorize_fp_with_semantic_features(event)
        fp_categories[category].append({
            'event': event,
            'score': score,
            'reason': reason,
            'features': extract_semantic_features_batch([event]).iloc[0].to_dict() if len(extract_semantic_features_batch([event])) > 0 else {}
        })

    # TEST 1: FEATURE QUALITY ASSESSMENT (CRITICAL)
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING SEMANTIC FEATURE INFORMATIVENESS")
    logger.info("="*80)

    all_events = benign_train + benign_test + malicious_events
    features_informative, feature_auc = test_feature_informativeness(all_events)

    # TEST 1.5: SIMPLE BASELINE COMPARISONS (NEW)
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING SIMPLE BASELINES (Before SAG)")
    logger.info("="*80)

    baseline_results = test_simple_baselines(all_events)

    # TEST 2: COMPUTATIONAL COST ESTIMATE
    logger.info("\n" + "="*80)
    logger.info("💰 COMPUTATIONAL COST ESTIMATE FOR SAG")
    logger.info("="*80)

    # Estimate average sequence length
    all_sequences = train_sequences + test_benign_sequences + test_malicious_sequences
    avg_seq_length = sum(len(seq) for seq in all_sequences) / len(all_sequences) if all_sequences else 50

    n_events = len(all_events)
    computationally_feasible = estimate_sag_computational_cost(n_events, avg_seq_length)

    # ANALYZE FP PATTERNS WITH COMPREHENSIVE TESTS
    fp_analysis = analyze_fp_patterns_with_tests(actual_fps, fp_categories, len(benign_test),
                                               features_informative, feature_auc,
                                               computationally_feasible)

    # SAVE DETAILED RESULTS
    output_dir = Path("experiments/phase1_5")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "detailed_fp_analysis.pkl", "wb") as f:
        pickle.dump({
            'fp_analysis': fp_analysis,
            'fp_categories': fp_categories,
            'actual_fps': actual_fps,
            'model_results': {
                'benign_scores': benign_scores,
                'malicious_scores': malicious_scores,
                'threshold': threshold,
                'benign_predictions': benign_predictions,
                'malicious_predictions': malicious_predictions
            },
            'tests': {
                'features_informative': features_informative,
                'feature_auc': feature_auc,
                'computationally_feasible': computationally_feasible
            },
            'n_benign_train': len(benign_train),
            'n_benign_test': len(benign_test),
            'n_malicious': len(malicious_events),
            'timestamp': datetime.now().isoformat()
        }, f)

    logger.info(f"✅ Saved detailed analysis to {output_dir}")

    # PRINT SUMMARY
    print_fp_analysis_summary(fp_analysis)

    # Clean up
    aggressive_cleanup(benign_events, malicious_events, correlated_events, ngram_model)

    return 0


def analyze_fp_patterns_with_tests(actual_fps, fp_categories, total_benign_test, features_informative, feature_auc, computationally_feasible):
    """
    Analyze patterns in actual false positives with comprehensive tests
    """
    logger.info("🔍 Analyzing patterns in actual false positives...")

    # Calculate statistics
    total_fps = len(actual_fps)
    category_counts = {k: len(v) for k, v in fp_categories.items()}

    # Calculate actual FP distribution
    fp_distribution = {k: len(v) for k, v in fp_categories.items()}

    # COMPREHENSIVE DECISION LOGIC (all tests must pass)
    # Extract best baseline AUC
    baseline_aucs = [r.get('auc', 0.5) for r in baseline_results.values() if 'auc' in r]
    best_baseline_auc = max(baseline_aucs) if baseline_aucs else 0.5

    should_proceed_sag = (
        total_fps / total_benign_test > 0.5 if total_benign_test > 0 else False and  # >50% of FPs are semantic gap
        features_informative and      # Features are informative enough
        computationally_feasible and   # SAG is computationally feasible
        best_baseline_auc < 0.85     # SAG justified only if baselines are not excellent
    )

    overall_fp_rate = total_fps / total_benign_test if total_benign_test > 0 else 0

    recommendations = []

    # Main decision
    if should_proceed_sag:
        recommendations.append("✅ PROCEED TO SAG - All tests passed!")
        recommendations.append(f"   ✓ FP rate: {overall_fp_rate*100:.1f}%")
        recommendations.append(f"   ✓ Feature AUC: {feature_auc:.3f}")
        recommendations.append(f"   ✓ Best baseline AUC: {best_baseline_auc:.3f}")
        recommendations.append("   ✓ Computationally feasible")
        recommendations.append("   ✓ SAG justified over simpler baselines")
    else:
        recommendations.append("⚠️ RECONSIDER SAG - One or more tests failed")

        if overall_fp_rate <= 0.5:
            recommendations.append(f"   ❌ FP rate too low: {overall_fp_rate*100:.1f}%")
            recommendations.append("   Consider simpler approaches first:")
            recommendations.append("   - Improve feature engineering")
            recommendations.append("   - Try context-aware n-grams")
            recommendations.append("   - Fix data quality issues")

        if not features_informative:
            recommendations.append(f"   ❌ Features not informative: AUC {feature_auc:.3f}")
            recommendations.append("   Critical: Current features don't capture semantic context")
            recommendations.append("   Solutions:")
            recommendations.append("   1. Improve feature engineering (consider LLM assistance)")
            recommendations.append("   2. Add more domain-specific features")
            recommendations.append("   3. Use end-to-end learning instead of SAG")

        if best_baseline_auc >= 0.85:
            recommendations.append(f"   ❌ Excellent baseline performance: AUC {best_baseline_auc:.3f}")
            recommendations.append("   SAG may not provide significant improvement")
            recommendations.append("   Consider if complexity is justified")

        if not computationally_feasible:
            recommendations.append("   ❌ Computationally infeasible for SAG")
            recommendations.append("   Solutions:")
            recommendations.append("   1. Reduce sequence length (max_events per session)")
            recommendations.append("   2. Use sparse attention (only key positions)")
            recommendations.append("   3. Approximate TreeSHAP")
            recommendations.append("   4. Try simpler methods first")

    # Overall FP rate assessment
    if overall_fp_rate < 0.01:  # <1% FP rate
        recommendations.append("✅ Overall FP rate is very low - current approach is good")
    elif overall_fp_rate > 0.10:  # >10% FP rate
        recommendations.append("❌ Overall FP rate is high - need significant improvement")

    # Data quality check
    if len(fp_categories['data_quality']) > 0.3 * total_fps:
        recommendations.append("⚠️ High data quality issues - fix data preprocessing first")

    return {
        'fp_categories': fp_categories,
        'category_counts': category_counts,
        'actual_fps': total_fps,
        'fp_distribution': fp_distribution,
        'total_benign_test': total_benign_test,
        'overall_fp_rate': overall_fp_rate,
        'should_proceed_sag': should_proceed_sag,
        'recommendations': recommendations,
        'feature_auc': feature_auc,
        'computationally_feasible': computationally_feasible
    }


def print_fp_analysis_summary(analysis):
    """Print a summary of the FP analysis with comprehensive tests"""
    logger.info("\n" + "="*80)
    logger.info("📊 COMPREHENSIVE FALSE POSITIVE ANALYSIS SUMMARY")
    logger.info("="*80)

    logger.info(f"📊 Total benign sessions tested: {analysis['total_benign_test']:,}")
    logger.info(f"📊 Actual false positives: {analysis['actual_fps']:,}")
    logger.info(f"📊 Actual FP rate: {analysis['overall_fp_rate']*100:.2f}%")
    logger.info(f"📊 Feature AUC: {analysis['feature_auc']:.3f}")
    logger.info(f"📊 SAG Computationally Feasible: {analysis['computationally_feasible']}")

    logger.info("\n📊 FP Category Breakdown:")
    for category, count in analysis['fp_distribution'].items():
        percentage = count / analysis['actual_fps'] * 100 if analysis['actual_fps'] > 0 else 0
        logger.info(f"  {category}: {count} ({percentage:.1f}%)")

    # Main decision
    logger.info("\n📋 RECOMMENDATION:")
    if analysis['should_proceed_sag']:
        logger.info("✅ PROCEED TO SAG - All tests passed!")
        logger.info(f"   ✓ FP rate: {analysis['overall_fp_rate']*100:.1f}%")
        logger.info(f"   ✓ Feature AUC: {analysis['feature_auc']:.3f}")
        logger.info("   ✓ Computationally feasible")
    else:
        logger.info("⚠️ RECONSIDER SAG - One or more tests failed")

        if analysis['overall_fp_rate'] <= 0.5:
            logger.info(f"   ❌ FP rate too low: {analysis['overall_fp_rate']*100:.1f}%")
        if analysis['feature_auc'] < 0.6:
            logger.info(f"   ❌ Features not informative: AUC {analysis['feature_auc']:.3f}")
        if not analysis['computationally_feasible']:
            logger.info("   ❌ Computationally infeasible")

        logger.info("   Consider simpler alternatives first")

    if analysis['overall_fp_rate'] < 0.01:
        logger.info("✅ Overall FP rate is very low - current approach is good")
    elif analysis['overall_fp_rate'] > 0.10:
        logger.info("❌ Overall FP rate is high - need improvement")


if __name__ == "__main__":
    sys.exit(run_detailed_fp_analysis())
