"""
Phase 0: Data Validation with Pandera Schemas
PRODUCTION READY - Run this after setup.sh passes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import pandera as pa
from datetime import datetime
from typing import Tuple  # ✅ ADDED for Python 3.8 compatibility
import gc  # ✅ ADDED for garbage collection
import psutil  # ✅ ADDED for memory monitoring
from src.data.lanl_loader import LANLLoader
from src.utils.reproducibility import set_seed

# Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"phase0_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ✅ FIXED: Schema matches actual LANL column names
class AuthEventSchema(pa.DataFrameModel):
    """Schema for LANL authentication events"""

    user_id: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="User identifier")
    src_comp_id: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Source computer")
    dst_comp_id: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Destination computer")
    auth_type: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Authentication type")
    outcome: pa.typing.String = pa.Field(description="Authentication outcome")
    timestamp: pa.typing.DateTime = pa.Field(description="Event timestamp")
    day: pa.typing.Int64 = pa.Field(ge=1, le=366, description="Day number")

    class Config:
        coerce = True
        strict = False  # ✅ Allow extra columns


class RedTeamEventSchema(pa.DataFrameModel):
    """Schema for red team attack events"""

    time: pa.typing.Int64 = pa.Field(ge=0, description="Attack timestamp (seconds)")
    user: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Target user")
    src_computer: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Source computer")
    dst_computer: pa.typing.String = pa.Field(str_length={"min_value": 1}, description="Destination computer")
    timestamp: pa.typing.DateTime = pa.Field(description="Attack timestamp")
    day: pa.typing.Int64 = pa.Field(ge=1, le=366, description="Day number")

    class Config:
        coerce = True
        strict = False


def validate_auth_data(df: pd.DataFrame) -> dict:
    """Validate authentication data"""
    logger.info("🔍 Validating authentication data...")

    results = {
        'total_events': len(df),
        'unique_users': df['user_id'].nunique(),
        'date_range': None,
        'schema_valid': False,
        'data_quality': {},
        'issues': []
    }

    if len(df) == 0:
        results['issues'].append("No authentication events found")
        return results

    # Date range
    try:
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        results['date_range'] = f"{min_date} to {max_date} ({(max_date - min_date).days} days)"
    except Exception as e:
        results['issues'].append(f"Date range calculation failed: {e}")

    # Schema validation
    try:
        AuthEventSchema.validate(df, lazy=True)
        results['schema_valid'] = True
        logger.info("✅ Authentication schema validation passed")
    except pa.errors.SchemaErrors as e:
        results['schema_valid'] = False
        results['issues'].append(f"Schema validation failed: {len(e.schema_errors)} errors")
        for error in list(e.schema_errors)[:3]:  # Show first 3 errors
            results['issues'].append(f"  - {error}")
        if len(e.schema_errors) > 3:
            results['issues'].append(f"  - ... and {len(e.schema_errors) - 3} more errors")
    except Exception as e:
        results['schema_valid'] = False
        results['issues'].append(f"Schema validation error: {e}")

    # Data quality checks
    quality = {}

    # User ID range
    if 'user_id' in df.columns:
        quality['user_id_stats'] = {
            'unique': int(df['user_id'].nunique()),
            'total_events': len(df)
        }

    # Auth types distribution
    if 'auth_type' in df.columns:
        quality['auth_types'] = df['auth_type'].value_counts().head(5).to_dict()

    # Outcome distribution
    if 'outcome' in df.columns:
        quality['outcomes'] = df['outcome'].value_counts().to_dict()

    results['data_quality'] = quality

    return results


def validate_redteam_data(df: pd.DataFrame) -> dict:
    """Validate red team data"""
    logger.info("🔍 Validating red team data...")

    results = {
        'total_events': len(df),
        'unique_users': df['user'].nunique() if 'user' in df.columns else 0,
        'schema_valid': False,
        'issues': []
    }

    if len(df) == 0:
        results['issues'].append("No red team events found")
        return results

    # Schema validation
    try:
        RedTeamEventSchema.validate(df, lazy=True)
        results['schema_valid'] = True
        logger.info("✅ Red team schema validation passed")
    except pa.errors.SchemaErrors as e:
        results['schema_valid'] = False
        results['issues'].append(f"Schema validation failed: {len(e.schema_errors)} errors")
        for error in list(e.schema_errors)[:3]:
            results['issues'].append(f"  - {error}")
    except Exception as e:
        results['schema_valid'] = False
        results['issues'].append(f"Schema validation error: {e}")

    return results


def log_memory_usage(stage: str):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1024 / 1024 / 1024
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"💾 Memory at {stage}: {mem_gb:.2f} GB (Available: {available_gb:.2f} GB)")
    except Exception as e:
        logger.warning(f"Could not log memory: {e}")


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


def check_sufficient_data(auth_df: pd.DataFrame, redteam_df: pd.DataFrame) -> Tuple[bool, str]:
    """Check if we have sufficient data for validation"""
    issues = []

    if len(auth_df) == 0:
        issues.append("No authentication events found")
    if len(redteam_df) == 0:
        issues.append("No red team events found")
    if len(redteam_df) < 5:
        issues.append(f"Very few red team events ({len(redteam_df)}) - may not be representative")

    return len(issues) == 0, "; ".join(issues) if issues else "Data sufficient"


def main():
    """Run Phase 0 validation"""
    set_seed(42)

    try:
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 0: DATA VALIDATION")
        logger.info("="*80)

        # Check if dataset exists
        data_dir = Path("data/raw/lanl")
        auth_file = data_dir / "auth.txt"
        auth_gz_file = data_dir / "auth.txt.gz"

        if not (auth_file.exists() or auth_gz_file.exists()):
            logger.error("❌ LANL dataset not found!")
            logger.error("📥 Please download from: https://csr.lanl.gov/data/cyber1/")
            return 1

        # Load data - use days with attacks for validation
        logger.info("\n📂 Loading LANL dataset...")

        # ✅ ULTRA-AGGRESSIVE: Check memory before ANY loading
        if not check_memory_or_abort("data_loading", min_gb=4.0):
            logger.error("❌ Insufficient memory for Phase 0 - need at least 4GB")
            return 1

        # ✅ ULTRA-AGGRESSIVE: Detect attack days with MINIMAL data
        logger.info("🔍 Detecting attack days (minimal memory)...")
        try:
            # Read ONLY first 1000 lines to detect days (was 10K)
            redteam_quick = pd.read_csv(
                data_dir / "redteam.txt",
                header=None,
                names=['time', 'user', 'src_computer', 'dst_computer'],
                nrows=1000  # ✅ ULTRA-LIMIT: Only 1K lines
            )

            # Use the loader to get proper start date and calculate days correctly
            log_memory_usage("before_quick_loader")
            quick_loader = LANLLoader(data_dir)
            start_date = quick_loader.start_date

            # Convert time to proper timestamps and days using the detected start date
            redteam_quick['timestamp'] = start_date + pd.to_timedelta(redteam_quick['time'], unit='s')
            redteam_quick['day'] = redteam_quick['timestamp'].dt.dayofyear
            attack_days = sorted(redteam_quick['day'].unique())

            # ✅ ULTRA-AGGRESSIVE: Use the first attack day
            if attack_days:
                validation_day = attack_days[0]
                logger.info(f"📅 Using attack day: {validation_day} (actual day {validation_day})")
            else:
                logger.error("❌ No attack days found")
                return 1

            # Clean up
            aggressive_cleanup(redteam_quick, quick_loader)

        except Exception as e:
            logger.error(f"❌ Failed to detect attack days: {e}")
            return 1

        # ✅ ULTRA-AGGRESSIVE: Check memory before loader
        if not check_memory_or_abort("before_loader", min_gb=3.0):
            logger.error("❌ Insufficient memory for loader creation")
            return 1

        log_memory_usage("before_loading")
        loader = LANLLoader(data_dir)

        # ✅ ULTRA-AGGRESSIVE: Load with SEVERE memory limits
        logger.info("📊 Loading with SEVERE memory limits...")
        # Use the flexible loader that loads auth from a window around attack day
        auth_df, redteam_df = loader.load_sample_flexible(
            attack_day=validation_day,
            auth_window_days=3,  # 3-day window around attack
            max_rows=500_000  # ✅ ULTRA-LIMIT: Max 500K events (was 2M)
        )
        log_memory_usage("after_loading")

        # ✅ ULTRA-AGGRESSIVE: Immediate cleanup of detection data
        if 'redteam_quick' in locals():
            aggressive_cleanup(redteam_quick)

        # ✅ ULTRA-AGGRESSIVE: Tiny sample for validation (was 10K, now 5K)
        if len(auth_df) > 5000:
            auth_df = auth_df.sample(n=5000, random_state=42)  # ✅ ULTRA-LIMIT: Max 5K events
            logger.info(f"✅ Sampled {len(auth_df):,} auth events for validation")

        # Check data sufficiency
        data_ok, data_message = check_sufficient_data(auth_df, redteam_df)
        if not data_ok:
            logger.warning(f"⚠️ Data sufficiency issues: {data_message}")

        # ✅ ULTRA-AGGRESSIVE: Memory check before validation
        if not check_memory_or_abort("before_auth_validation", min_gb=1.0):
            logger.error("❌ Insufficient memory for authentication validation")
            return 1

        # Validate
        logger.info("\n" + "="*60)
        logger.info("AUTHENTICATION DATA VALIDATION")
        logger.info("="*60)
        auth_results = validate_auth_data(auth_df)

        logger.info(f"\n📊 Authentication Data Summary:")
        logger.info(f"  Total events: {auth_results['total_events']:,}")
        logger.info(f"  Unique users: {auth_results['unique_users']:,}")
        if auth_results['date_range']:
            logger.info(f"  Date range: {auth_results['date_range']}")

        if auth_results['schema_valid']:
            logger.info("✅ Schema validation: PASSED")
        else:
            logger.error("❌ Schema validation: FAILED")
            for issue in auth_results['issues']:
                logger.error(f"  {issue}")

        # ✅ ULTRA-AGGRESSIVE: Memory check before red team validation
        if not check_memory_or_abort("before_redteam_validation", min_gb=0.5):
            logger.error("❌ Insufficient memory for red team validation")
            return 1

        logger.info("\n" + "="*60)
        logger.info("RED TEAM DATA VALIDATION")
        logger.info("="*60)
        redteam_results = validate_redteam_data(redteam_df)

        logger.info(f"\n📊 Red Team Data Summary:")
        logger.info(f"  Total events: {redteam_results['total_events']:,}")
        logger.info(f"  Unique users: {redteam_results['unique_users']:,}")

        if redteam_results['schema_valid']:
            logger.info("✅ Schema validation: PASSED")
        else:
            logger.error("❌ Schema validation: FAILED")
            for issue in redteam_results['issues']:
                logger.error(f"  {issue}")

        # ✅ ULTRA-AGGRESSIVE: Memory check before final decision
        if not check_memory_or_abort("before_final_decision", min_gb=0.5):
            logger.error("❌ Insufficient memory for final decision")
            return 1

        # Overall
        all_passed = (
            auth_results['schema_valid'] and
            redteam_results['schema_valid'] and
            len(auth_df) > 0 and
            len(redteam_df) > 0 and
            data_ok  # ✅ ADDED: Check data sufficiency
        )

        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("✅ VALIDATION PASSED - Ready for Phase 1")
            return_code = 0
        else:
            logger.error("❌ VALIDATION FAILED - Fix issues above")
            return_code = 1

        # ✅ ULTRA-AGGRESSIVE: Comprehensive cleanup (always executes)
        log_memory_usage("before_cleanup")
        aggressive_cleanup(auth_df, redteam_df, redteam_quick, loader)
        # Force multiple garbage collections
        for _ in range(3):
            gc.collect()
        log_memory_usage("after_cleanup")

        return return_code

    except Exception as e:
        logger.error(f"\n❌ Phase 0 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
