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
from src.utils.reproducibility import set_seed
from src.utils.memory_monitor import MemoryMonitor, memory_safe  # ✅ NEW

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

        # Detect attack days
        logger.info("🔍 Detecting attack days...")
        try:
            redteam_quick = pd.read_csv(redteam_file, header=None,
                                       names=['time', 'user', 'src_computer', 'dst_computer'])
            redteam_quick['day'] = (redteam_quick['time'] / 86400).astype(int) + 1
        except:
            redteam_quick = pd.read_csv(redteam_file, header=None,
                                       names=['timestamp', 'user_id', 'action'])
            redteam_quick['timestamp'] = pd.to_datetime(redteam_quick['timestamp'])
            redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear

        attack_days = sorted(redteam_quick['day'].unique())
        logger.info(f"📅 Attack days: {attack_days}")

        # ✅ Load full attack campaign for statistical power (7-9 days)
        days_to_load = list(range(
            attack_days[0] - 2,
            min(attack_days[-1] + 3, attack_days[0] + 9)  # Max 9 days for pilot
        ))

        logger.info(f"📊 Loading {len(days_to_load)} days: {days_to_load[0]}-{days_to_load[-1]}")
        logger.info("   (Optimized for 62GB system - ~15M events expected)")

        auth_df, redteam_df = loader.load_sample(days=days_to_load)

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

        # Build sessions
        logger.info("\n🔧 Building sessions...")
        config = SessionConfig(
            timeout_minutes=30,
            min_events=3,
            max_events=100,  # ✅ Increased for better analysis
            labeling="window",
            label_window_minutes=240
        )
        builder = SessionBuilder(config)

        all_sessions = builder.build_sessions(auth_df, redteam_df, train_mode=False)

        # ✅ Clean up auth_df immediately
        del auth_df, redteam_df
        gc.collect()
        monitor.log_usage("After session building")

        benign = [s for s in all_sessions if not s['is_malicious']]
        malicious = [s for s in all_sessions if s['is_malicious']]

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

        # ✅ SCALED for 62GB system (optimized usage)
        MAX_BENIGN = 30_000
        if len(benign) > MAX_BENIGN:
            logger.warning(f"⚠️ Sampling benign: {len(benign)} → {MAX_BENIGN}")
            random.seed(42)
            benign = random.sample(benign, MAX_BENIGN)
            gc.collect()

        monitor.log_usage("After session filtering")

        # ✅ ADDED: Memory monitoring before analysis
        if not check_memory_or_abort("context_analysis", min_gb=1.0):
            return 1

        # Run analysis
        logger.info("\n🔬 Running context analysis...")
        analyzer = ContextWindowAnalyzer(
            n_values=[1, 2, 3, 5, 10],  # ✅ Include n=5, 10 for richer context
            cv_folds=5  # ✅ 5-fold CV for reliability
        )
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

        # ✅ ADDED: Final cleanup
        logger.info("🧹 Cleaning up memory...")
        del benign, malicious, all_sessions
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
