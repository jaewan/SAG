"""
Phase 1: Context Window Analysis
PRODUCTION READY - Run this after phase0_validate.py passes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pickle
import pandas as pd
from datetime import datetime
from src.data.lanl_loader import LANLLoader
from src.data.session_builder import SessionBuilder, SessionConfig
from src.evaluation.context_analysis import ContextWindowAnalyzer
from src.utils.reproducibility import set_seed

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


def main():
    """Run Phase 1"""
    set_seed(42)

    try:
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 1: CONTEXT WINDOW ANALYSIS")
        logger.info("="*80)

        # Load data
        logger.info("\n📂 Loading LANL dataset...")
        loader = LANLLoader(Path("data/raw/lanl"))

        # STEP 1: Load redteam labels first (to detect attack days)
        redteam_file = Path("data/raw/lanl/redteam.txt")
        if not redteam_file.exists():
            logger.error("❌ No red team labels!")
            logger.error(" Check: data/raw/lanl/redteam.txt")
            return 1

        # Quick load of redteam to get days
        logger.info("🔍 Detecting attack days from redteam file...")
        try:
            # Try loading as CSV with headers (for real LANL data)
            redteam_quick = pd.read_csv(redteam_file)
            if 'time' in redteam_quick.columns:
                # Real LANL format: time (seconds), user, src, dst
                redteam_quick['day'] = (redteam_quick['time'] / 86400).astype(int) + 1
            else:
                # Synthetic format: timestamp, user, action
                redteam_quick['timestamp'] = pd.to_datetime(redteam_quick['timestamp'], format='%m/%d/%Y %H:%M:%S')
                redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear
        except:
            # Fallback: assume synthetic format
            redteam_quick = pd.read_csv(redteam_file, header=None, names=['timestamp', 'user_id', 'action'])
            redteam_quick['timestamp'] = pd.to_datetime(redteam_quick['timestamp'], format='%m/%d/%Y %H:%M:%S')
            redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear

        attack_days = sorted(redteam_quick['day'].unique())

        logger.info(f"📅 Detected attack days: {attack_days}")

        # STEP 2: Load those days + buffer
        days_to_load = list(range(attack_days[0] - 1, attack_days[-1] + 2))
        logger.info(f"📊 Loading days: {days_to_load[0]}-{days_to_load[-1]}")

        auth_df, redteam_df = loader.load_sample(days=days_to_load)

        # Validation
        if len(redteam_df) == 0:
            logger.error("❌ No red team events found in loaded days!")
            return 1

        logger.info(f"✅ Loaded {len(auth_df):,} auth events, {len(redteam_df)} attacks")

        # Build sessions
        logger.info("\n🔧 Building sessions...")
        config = SessionConfig(
            timeout_minutes=30,
            min_events=3,
            labeling="window",
            label_window_minutes=240  # Wider window for better attack detection
        )
        builder = SessionBuilder(config)

        # Session timeout validation is handled internally by SessionBuilder

        all_sessions = builder.build_sessions(auth_df, redteam_df, train_mode=False)

        benign = [s for s in all_sessions if not s['is_malicious']]
        malicious = [s for s in all_sessions if s['is_malicious']]

        logger.info(f"\n📊 Dataset:")
        logger.info(f" Benign: {len(benign)}")
        logger.info(f" Malicious: {len(malicious)}")

        # Warning for small samples
        if len(malicious) < 15:
            logger.warning("\n⚠️ Very few malicious samples!")
            logger.warning(" Results may be noisy")
            logger.warning(" Proceeding anyway for demonstration purposes...")
            # Auto-proceed in non-interactive mode
            # response = input("\nProceed anyway? [y/N]: ").strip().lower()
            # if response != 'y':
            #     logger.info("Cancelled by user")
            #     return 0

        # Run analysis
        logger.info("\n🔬 Running context analysis...")
        analyzer = ContextWindowAnalyzer()
        results, decision = analyzer.analyze(benign, malicious)

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

        if decision == "proceed":
            logger.info("\n🎉 Ready for Phase 2!")
            return 0
        elif decision == "proceed_caution":
            logger.info("\n⚠️ Can proceed to Phase 2 (results may be marginal)")
            return 0
        else:
            logger.info("\n❌ Consider alternative approaches")
            return 1

    except KeyboardInterrupt:
        logger.info("\n\n❌ Cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
