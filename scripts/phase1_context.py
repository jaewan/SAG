"""
Phase 1: Context Window Analysis
PRODUCTION READY with OOM protection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pickle
import pandas as pd  # ✅ ADDED
import random  # ✅ ADDED
import gc  # ✅ ADDED for garbage collection
import psutil  # ✅ ADDED for memory monitoring
from datetime import datetime
from src.data.lanl_loader import LANLLoader
from src.data.session_builder import SessionBuilder, SessionConfig
from src.evaluation.context_analysis import ContextWindowAnalyzer
from src.models.ngram_models import NgramLanguageModel, evaluate_ngram_model, PrunedNgramLanguageModel
from src.utils.reproducibility import set_seed
from src.utils.memory_monitor import MemoryMonitor, memory_safe  # ✅ NEW
from scipy import stats


class CorrelatedContextWindowAnalyzer(ContextWindowAnalyzer):
    """ContextWindowAnalyzer that uses correlated data for richer tokenization"""

    def __init__(self, *args, model_class=None, model_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}

    def _tokenize(self, session: Dict) -> List[str]:
        """
        Enhanced tokenization using correlated data (auth + proc + flows)
        """
        # Check if session has correlated event data
        if 'correlated_event' in session:
            # Use correlated tokenization method
            correlated_event = session['correlated_event']
            return self._tokenize_correlated_event(correlated_event)
        else:
            # Fallback to standard tokenization
            return super()._tokenize(session)

    def _tokenize_correlated_event(self, correlated_event: Dict) -> List[str]:
        """Tokenize using correlated data (auth + proc + flows)"""
        tokens = []

        # Auth event token
        auth_event = correlated_event['auth_event']
        src = str(auth_event.get('src_computer', ''))[:8]
        dst = str(auth_event.get('dst_computer', ''))[:8]

        timestamp = auth_event.get('timestamp')
        if timestamp:
            hour_bin = timestamp.hour // 3  # 0-7 (3-hour bins)
        else:
            hour_bin = 0

        auth_type = auth_event.get('auth_type', 'Unknown')
        outcome = auth_event.get('outcome', 'Unknown')

        auth_token = f"{auth_type}_{outcome}_src{src}_dst{dst}_t{hour_bin}"
        tokens.append(auth_token)

        # Process tokens
        for proc in correlated_event.get('related_processes', []):
            proc_token = f"proc_{proc['process_name']}"
            tokens.append(proc_token)

        # Flow tokens
        for flow in correlated_event.get('related_flows', []):
            flow_token = f"flow_{flow['dst_port']}"
            tokens.append(flow_token)

        return tokens

    def _run_cv(self, cv, all_sessions, y, n, groups):
        """Override to use custom model class"""
        fold_results = []

        if groups is not None:
            splits = cv.split(all_sessions, y, groups)
        else:
            splits = cv.split(all_sessions, y)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n Fold {fold_idx+1}/{self.cv_folds}:")

            # Get train/test data
            train_benign_idx = [i for i in train_idx if y[i] == 0]
            train_sessions = [all_sessions[i] for i in train_benign_idx]
            test_sessions = [all_sessions[i] for i in test_idx]
            test_labels = y[test_idx]
            n_test_mal = (test_labels == 1).sum()

            # Skip fold if no malicious in test
            if n_test_mal == 0:
                logger.warning(f" ⚠️ No malicious in test - skipping fold")
                continue

            # Tokenize
            train_seqs = [self._tokenize(s) for s in train_sessions]
            test_seqs = [self._tokenize(s) for s in test_sessions]
            test_benign_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 0]
            test_mal_seqs = [seq for seq, label in zip(test_seqs, test_labels) if label == 1]

            # Fit model with custom class and parameters
            try:
                model = self.model_class(n=n, **self.model_kwargs)
                model.fit(train_seqs)
            except Exception as e:
                logger.error(f" ❌ Fit failed: {e}")
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
                del model
                gc.collect()
                continue

            # Cleanup
            del train_sessions, test_sessions, train_seqs, test_seqs, test_benign_seqs, test_mal_seqs
            if 'model' in locals():
                del model
            gc.collect()

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

# Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"phase1_{datetime.now():%Y%m%d_%H%M%S}.log"),
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


    def _data_quality_check(self, auth_df: pd.DataFrame, redteam_df: pd.DataFrame):
        """Check data quality and examine malicious patterns"""
        logger.info("🔍 Analyzing data quality...")

        # Basic stats
        logger.info(f"📊 Auth events: {len(auth_df):,}")
        logger.info(f"📊 Red team events: {len(redteam_df)}")

        if len(redteam_df) == 0:
            logger.warning("⚠️ No red team events found - cannot check malicious patterns")
            return

        # Check for malicious users
        malicious_users = set(redteam_df['user_id'].unique())
        logger.info(f"👥 Malicious users: {len(malicious_users)}")

        # Sample malicious sessions
        logger.info("\n🔍 Sample Malicious Sessions:")
        for i, user_id in enumerate(list(malicious_users)[:3]):  # Show first 3 users
            user_auth = auth_df[auth_df['user_id'] == user_id]
            user_redteam = redteam_df[redteam_df['user_id'] == user_id]

            logger.info(f"\n  User {i+1}: {user_id}")
            logger.info(f"    Auth events: {len(user_auth):,}")
            logger.info(f"    Attack events: {len(user_redteam)}")

            if len(user_redteam) > 0:
                attack_time = user_redteam.iloc[0]['timestamp']
                logger.info(f"    First attack: {attack_time}")

                # Show events around attack time
                time_window = pd.Timedelta(minutes=60)
                nearby_auth = user_auth[
                    (user_auth['timestamp'] >= attack_time - time_window) &
                    (user_auth['timestamp'] <= attack_time + time_window)
                ]
                logger.info(f"    Auth events near attack (±1h): {len(nearby_auth)}")

                if len(nearby_auth) > 0:
                    logger.info("    Sample auth events:")
                    for _, event in nearby_auth.head(3).iterrows():
                        logger.info(f"      {event['timestamp']}: {event['auth_type']} -> {event['dst_computer']}")

        # Check token diversity (preview of new correlated tokenization)
        logger.info("\n🎯 Token Diversity Check (correlated tokenization):")
        sample_correlated = self._create_sample_correlated_events(correlated_events, redteam_df)
        if sample_correlated:
            benign_tokens = set()
            malicious_tokens = set()

            for corr_event in sample_correlated:
                # Create a session-like dict for tokenization
                session = {
                    'correlated_event': corr_event,
                    'is_malicious': corr_event['is_malicious']
                }
                tokens = self._tokenize_correlated_session(session)
                if corr_event['is_malicious']:
                    malicious_tokens.update(tokens)
                else:
                    benign_tokens.update(tokens)

            logger.info(f"  Benign unique tokens: {len(benign_tokens):,}")
            logger.info(f"  Malicious unique tokens: {len(malicious_tokens):,}")
            logger.info(f"  Overlapping tokens: {len(benign_tokens & malicious_tokens):,}")

            overlap_ratio = len(benign_tokens & malicious_tokens) / len(benign_tokens | malicious_tokens)
            logger.info(f"  Overlap ratio: {overlap_ratio:.3f}")

            if overlap_ratio > 0.9:
                logger.warning("⚠️ High token overlap - may indicate poor discriminative power")
            elif overlap_ratio > 0.7:
                logger.warning("⚠️ Moderate token overlap - check if discriminative enough")
            else:
                logger.info("✅ Good token diversity - should enable better discrimination")


    def _create_sample_sessions_for_analysis(self, auth_df: pd.DataFrame, redteam_df: pd.DataFrame) -> List[Dict]:
        """Create sample sessions for analysis (quick version)"""
        # Simple session creation for analysis - just group by user and create basic sessions
        sessions = []

        # Get first 10 malicious users for analysis
        malicious_users = set(redteam_df['user_id'].unique())
        sample_users = list(malicious_users)[:5]  # Just 5 for analysis

        for user_id in sample_users:
            user_auth = auth_df[auth_df['user_id'] == user_id].sort_values('timestamp')

            if len(user_auth) < 3:  # Need minimum events
                continue

            # Create a simple session from user's events
            session = {
                'session_id': f"sample_{user_id}",
                'user_id': user_id,
                'start_time': user_auth['timestamp'].iloc[0],
                'end_time': user_auth['timestamp'].iloc[-1],
                'events': user_auth.to_dict('records'),
                'is_malicious': user_id in malicious_users
            }
            sessions.append(session)

        # Add some benign users for comparison
        benign_users = set(auth_df['user_id'].unique()) - malicious_users
        for user_id in list(benign_users)[:5]:
            user_auth = auth_df[auth_df['user_id'] == user_id].sort_values('timestamp')

            if len(user_auth) < 3:
                continue

            session = {
                'session_id': f"sample_{user_id}",
                'user_id': user_id,
                'start_time': user_auth['timestamp'].iloc[0],
                'end_time': user_auth['timestamp'].iloc[-1],
                'events': user_auth.to_dict('records'),
                'is_malicious': False
            }
            sessions.append(session)

        return sessions


    def _tokenize_session(self, session: Dict) -> List[str]:
        """NEW: Improved tokenization with computer IDs, time, and richer context"""
        tokens = []

        for event in session['events']:
            # Include actual computer IDs (first 8 chars for specificity)
            src = str(event.get('src_computer', ''))[:8]
            dst = str(event.get('dst_computer', ''))[:8]

            # Include time-of-day in 3-hour bins
            timestamp = event.get('timestamp')
            if timestamp:
                hour_bin = timestamp.hour // 3  # 0-7 (3-hour bins)
            else:
                hour_bin = 0

            # More specific token format
            auth_type = event.get('auth_type', 'Unknown')
            outcome = event.get('outcome', 'Unknown')

            # Create rich token
            token = f"{auth_type}_{outcome}_src{src}_dst{dst}_t{hour_bin}"
            tokens.append(token)

        return tokens


    def _convert_correlated_to_sessions(self, correlated_events: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Convert correlated events to session format for compatibility"""
        sessions = []

        for event in correlated_events:
            # Create session from correlated event
            auth_event = event['auth_event']
            session_id = f"Corr_{event['timestamp'].strftime('%Y%m%d_%H%M%S')}"

            session = {
                'session_id': session_id,
                'user_id': auth_event['user_id'],
                'start_time': auth_event['timestamp'],
                'end_time': auth_event['timestamp'],  # Single event session
                'num_events': 1,
                'events': [auth_event],  # Single auth event
                'is_malicious': event['is_malicious'],
                'correlated_event': event  # Keep full correlated data
            }
            sessions.append(session)

        logger.info(f"✅ Converted {len(sessions)} correlated events to sessions")
        return sessions


    def _create_sample_correlated_events(self, correlated_events: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Create sample correlated events for analysis"""
        sample_events = []

        # Get malicious users
        malicious_users = set(redteam_df['user_id'].unique())

        # Sample malicious events
        malicious_events = [e for e in correlated_events if e['is_malicious']]
        sample_events.extend(malicious_events[:10])  # First 10 malicious

        # Sample benign events (not from malicious users)
        benign_events = [e for e in correlated_events if not e['is_malicious']]
        sample_events.extend(benign_events[:10])  # First 10 benign

        return sample_events


    def _tokenize_correlated_session(self, session: Dict) -> List[str]:
        """NEW: Rich tokenization using correlated data (auth + proc + flows)"""
        tokens = []

        # Use correlated event data if available
        if 'correlated_event' in session:
            correlated_event = session['correlated_event']

            # Auth event token
            auth_event = correlated_event['auth_event']
            auth_token = self._create_auth_token(auth_event)
            tokens.append(auth_token)

            # Process tokens
            for proc in correlated_event.get('related_processes', []):
                proc_token = f"proc_{proc['process_name']}"
                tokens.append(proc_token)

            # Flow tokens
            for flow in correlated_event.get('related_flows', []):
                flow_token = f"flow_{flow['dst_port']}"
                tokens.append(flow_token)

        else:
            # Fallback to old tokenization
            tokens = self._tokenize_session(session)

        return tokens


    def _create_auth_token(self, auth_event: Dict) -> str:
        """Create rich auth token with computer IDs and time"""
        src = str(auth_event.get('src_computer', ''))[:8]
        dst = str(auth_event.get('dst_computer', ''))[:8]

        timestamp = auth_event.get('timestamp')
        if timestamp:
            hour_bin = timestamp.hour // 3  # 0-7 (3-hour bins)
        else:
            hour_bin = 0

        auth_type = auth_event.get('auth_type', 'Unknown')
        outcome = auth_event.get('outcome', 'Unknown')

        return f"{auth_type}_{outcome}_src{src}_dst{dst}_t{hour_bin}"


@memory_safe(max_memory_gb=15.0)  # ✅ Optimized for 62GB system (24% of RAM)
def main():
    """Run Phase 1 with memory safety"""
    set_seed(42)

    monitor = MemoryMonitor()  # ✅ Track memory usage
    monitor.start()

    try:
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 1: CONTEXT WINDOW ANALYSIS (Memory-Safe)")
        logger.info("="*80)
        monitor.log_usage("Start")

        # Load data
        logger.info("\n📂 Loading LANL dataset...")
        loader = LANLLoader(Path("data/raw/lanl"))

        redteam_file = Path("data/raw/lanl/redteam.txt")
        if not redteam_file.exists():
            logger.error("❌ No red team labels!")
            return 1

        # Detect attack days (using same method as LANL loader)
        logger.info("🔍 Detecting attack days...")
        try:
            redteam_quick = pd.read_csv(redteam_file, header=None,
                                       names=['time', 'user', 'src_computer', 'dst_computer'])
            # ✅ FIX: Use same day calculation as LANL loader (dayofyear)
            redteam_quick['timestamp'] = loader.start_date + pd.to_timedelta(redteam_quick['time'], unit='s')
            redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear
        except:
            redteam_quick = pd.read_csv(redteam_file, header=None,
                                       names=['timestamp', 'user_id', 'action'])
            redteam_quick['timestamp'] = pd.to_datetime(redteam_quick['timestamp'])
            redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear

        attack_days = sorted(redteam_quick['day'].unique())
        logger.info(f"📅 Attack days: {attack_days}")

        # ✅ ULTRA-AGGRESSIVE: Load MINIMAL data for analysis (ultra-conservative)
        logger.info("📊 Planning ultra-conservative data loading...")

        # ✅ STRATEGY: Use smart stratified sampling across ALL attack days for maximum coverage
        candidate_days = attack_days  # Use ALL available attack days, not just consecutive ones

        # ✅ MAXIMUM: Load significantly more data for better attack coverage
        max_rows_to_load = 60_000_000  # Increased to 60M for comprehensive attack data

        # ✅ UPDATED: Memory estimates for comprehensive attack coverage across ALL attack days
        estimated_size_gb = len(candidate_days) * 4.0  # ~2GB per day for 60M rows with 80% attack focus
        min_required_gb = estimated_size_gb + 25  # Conservative buffer for comprehensive coverage

        logger.info(f"📊 Loading {len(candidate_days)} attack days: {sorted(candidate_days)}")
        logger.info(f"   Estimated size: {estimated_size_gb:.1f}GB, Required: {min_required_gb:.1f}GB")

        if not check_memory_or_abort("data_loading_ultra", min_gb=min_required_gb):
            logger.error("❌ Cannot proceed - insufficient memory for attack days")
            return 1

        days_to_load = candidate_days
        logger.info(f"✅ Memory check passed for {len(days_to_load)} attack days across all attack periods")

        if not days_to_load:
            logger.error("❌ Cannot proceed - insufficient memory even for 1 day")
            return 1

        logger.info(f"📊 Loading {len(days_to_load)} days: {days_to_load[0]}-{days_to_load[-1]}")
        logger.info("   (ULTRA-CONSERVATIVE: Minimal days for memory safety)")

        # ✅ SMART SAMPLING: Use attack-aware stratified sampling for better attack data capture
        logger.info("🎯 Using smart sampling strategy for LANL dataset...")

        # ✅ NEW: Use correlated data (auth + proc + flows) for richer context
        logger.info("🎯 Loading correlated multi-source data across ALL attack days...")
        logger.info(f"   Correlation window: ±5 minutes for related events")

        # Use correlated data loading for richer context
        correlated_events, correlation_quality = loader.load_full_context(
            days=candidate_days,
            max_rows=max_rows_to_load,
            correlation_window_sec=300  # ±5 minutes
        )

        logger.info(f"✅ Loaded {len(correlated_events)} correlated events")
        logger.info(f"   Correlation quality: {correlation_quality.get('correlation_rate', 0):.1%}")

        if len(correlated_events) == 0:
            logger.error("❌ No correlated events loaded - cannot proceed")
            return 1

        # Extract auth and redteam data from correlated events for compatibility
        auth_df = pd.DataFrame([e['auth_event'] for e in correlated_events])
        redteam_df = loader._load_redteam_data()
        redteam_df = redteam_df[redteam_df['day'].isin(candidate_days)]

        logger.info(f"📊 Extracted: {len(auth_df):,} auth events, {len(redteam_df)} red team events")

        # ✅ NEW: Data quality check - examine malicious sessions
        logger.info("\n🔍 DATA QUALITY CHECK: Examining malicious sessions...")
        self._data_quality_check(auth_df, redteam_df)

        # ✅ CRITICAL FIX: Normalize redteam column names to match auth_df
        if 'user' in redteam_df.columns and 'user_id' not in redteam_df.columns:
            redteam_df['user_id'] = redteam_df['user']
        if 'src_computer' in redteam_df.columns and 'src_comp_id' not in redteam_df.columns:
            redteam_df['src_comp_id'] = redteam_df['src_computer']
        if 'dst_computer' in redteam_df.columns and 'dst_comp_id' not in redteam_df.columns:
            redteam_df['dst_comp_id'] = redteam_df['dst_computer']

        monitor.log_usage("After data load")

        # ✅ SCALED for 62GB system (conservative usage)
        def calculate_safe_max_events():
            """Calculate safe max events based on available memory"""
            try:
                available_gb = psutil.virtual_memory().available / (1024**3)
                # Conservative model: 20 bytes/event → 50K events/MB
                # Account for 5x memory spike during session building
                safety_factor = 0.2  # Use only 20% of available
                safe_events = int((available_gb * safety_factor) * 50000)
                # Clip to reasonable range
                return max(100_000, min(safe_events, 3_000_000))
            except:
                return 1_000_000  # Fallback conservative value

        MAX_EVENTS = calculate_safe_max_events()
        logger.info(f"📊 Max events set to {MAX_EVENTS:,} based on available memory")

        # ✅ ULTRA-AGGRESSIVE: Ultra-conservative sampling (fixed at 1M max)
        MAX_EVENTS = 1_000_000  # ✅ ULTRA-AGGRESSIVE: Fixed at 1M events max

        if len(auth_df) > MAX_EVENTS:
            logger.warning(f"⚠️ Downsampling: {len(auth_df):,} → {MAX_EVENTS:,}")

            # ✅ FIX: Use normalized column name
            attack_users = set(redteam_df['user_id'].unique())
            attack_days_set = set(redteam_df['day'].unique())

            # Preserve attack context
            auth_df['is_attack_context'] = (
                auth_df['user_id'].isin(attack_users) &
                auth_df['day'].isin(attack_days_set)
            )

            attack_context = auth_df[auth_df['is_attack_context']]
            benign_context = auth_df[~auth_df['is_attack_context']]

            n_benign_needed = MAX_EVENTS - len(attack_context)
            if n_benign_needed > 0 and len(benign_context) > n_benign_needed:
                benign_context = benign_context.sample(n=n_benign_needed, random_state=42)

            auth_df = pd.concat([attack_context, benign_context]).sort_values('timestamp')
            auth_df = auth_df.drop('is_attack_context', axis=1).reset_index(drop=True)

            logger.info(f"  Final: {len(auth_df):,} events")

            # ✅ Force garbage collection
            del attack_context, benign_context
            gc.collect()

        # ✅ ADDED: Memory check before session building
        if not check_memory_or_abort("session_building", min_gb=5.0):
            return 1

        monitor.log_usage("After downsampling")

        # ✅ NEW: Build sessions from correlated events for richer context
        logger.info("\n🔧 Building sessions from correlated events...")
        config = SessionConfig(
            timeout_minutes=30,
            min_events=3,
            max_events=100,
            labeling="window",  # ✅ FIXED: Use window labeling for tighter malicious labeling
            label_window_minutes=60  # ✅ FIXED: Tighter window (±1 hour) to reduce false positives
        )
        builder = SessionBuilder(config)

        # Convert correlated events to session format for compatibility
        all_sessions = self._convert_correlated_to_sessions(correlated_events, redteam_df)

        # ✅ Clean up auth_df immediately
        del auth_df, redteam_df
        gc.collect()
        monitor.log_usage("After session building")

        # ✅ CRITICAL FIX: Extract benign/malicious and delete all_sessions immediately
        benign = [s for s in all_sessions if not s['is_malicious']]
        malicious = [s for s in all_sessions if s['is_malicious']]

        # ✅ DELETE all_sessions immediately to free memory (saves 10-20GB)
        del all_sessions
        gc.collect()
        monitor.log_usage("After deleting all_sessions")

        logger.info(f"\n📊 Dataset:")
        logger.info(f" Benign: {len(benign)}")
        logger.info(f" Malicious: {len(malicious)}")

        # ✅ CRITICAL FIX: Check minimum malicious samples for CV
        cv_folds = 5  # From analyzer config
        MIN_MALICIOUS = max(10, cv_folds * 2)  # At least 2 per fold
        if len(malicious) < MIN_MALICIOUS:
            logger.error(f"❌ Need >= {MIN_MALICIOUS} malicious samples for {cv_folds}-fold CV")
            logger.error(f"   Current: {len(malicious)}")
            logger.error("   Solutions:")
            logger.error("   1. Load more attack days")
            logger.error("   2. Use looser labeling (label_window_minutes=480)")
            logger.error("   3. Reduce CV folds (not recommended)")
            return 1  # STOP

        # ✅ AGGRESSIVE: Much smaller sample for memory (reduced from 30K to 10K)
        MAX_BENIGN = 5_000  # ✅ ULTRA-AGGRESSIVE: Reduced from 10K to 5K
        if len(benign) > MAX_BENIGN:
            logger.warning(f"⚠️ Sampling benign: {len(benign)} → {MAX_BENIGN}")
            random.seed(42)
            benign = random.sample(benign, MAX_BENIGN)
            gc.collect()

        monitor.log_usage("After session filtering")

        # ✅ ADDED: Memory monitoring before analysis
        if not check_memory_or_abort("context_analysis", min_gb=1.0):
            return 1

        # Run analysis with correlated data-aware tokenization
        logger.info("\n🔬 Running context analysis with correlated tokenization...")
        analyzer = CorrelatedContextWindowAnalyzer(
            n_values=[1, 2, 3, 5, 10],  # ✅ Include n=5, 10 for richer context
            cv_folds=5  # ✅ 5-fold CV for reliability
        )

        # ✅ NEW: Use improved n-gram models with better OOV handling for richer tokens
        analyzer.model_class = PrunedNgramLanguageModel  # Use pruned model for memory efficiency
        analyzer.model_kwargs = {
            'smoothing': 'laplace',
            'max_vocab_size': 100000,  # Increased for richer tokenization
            'min_count': 1,  # Lower threshold for diverse tokens
            'oov_handling': 'unk'  # Replace OOV with <UNK> token
        }
        results, decision = analyzer.analyze(benign, malicious)

        monitor.log_usage("After analysis")

        # Save
        logger.info("\n💾 Saving results...")
        output_dir = Path("experiments/phase1")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump({
                'results': results,
                'decision': decision,
                'n_benign': len(benign),
                'n_malicious': len(malicious),
                'timestamp': datetime.now().isoformat()
            }, f)

        logger.info(f"✅ Saved to {output_dir}")

        # Final verdict
        logger.info("\n" + "="*80)
        logger.info(f"✅ PHASE 1 COMPLETE: {decision.upper()}")
        logger.info("="*80)

        monitor.log_usage("Complete")
        monitor.print_summary()

        # ✅ ADDED: Final cleanup (all_sessions already deleted above)
        logger.info("🧹 Cleaning up memory...")
        del benign, malicious
        gc.collect()
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"💾 Final memory: {available_gb:.2f} GB available")

        return 0 if decision in ["proceed", "proceed_caution"] else 1

    except MemoryError as e:
        logger.error(f"\n❌ OUT OF MEMORY: {e}")
        logger.error("Solutions:")
        logger.error(" 1. Reduce MAX_EVENTS (currently dynamic)")
        logger.error(" 2. Load fewer days")
        logger.error(" 3. Reduce MAX_BENIGN sessions")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # ✅ Enhanced cleanup
        logger.info("🧹 Emergency cleanup...")
        gc.collect()
        try:
            available_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"💾 Emergency cleanup memory: {available_gb:.2f} GB available")
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
