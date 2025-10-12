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

    time: pa.typing.Int64 = pa.Field(ge=0, description="Event timestamp (seconds)")
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

    time: pa.typing.Int = pa.Field(ge=0, description="Attack timestamp (seconds)")
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
        'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else df['user'].nunique() if 'user' in df.columns else 0,
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

        # Check memory before loading
        if not check_memory_or_abort("data_loading", min_gb=3.0):
            return 1

        # First, quickly check what days have attacks
        redteam_quick = pd.read_csv(
            data_dir / "redteam.txt",
            header=None,
            names=['time', 'user', 'src_computer', 'dst_computer']
        )
        redteam_quick['day'] = (redteam_quick['time'] / 86400).astype(int) + 1
        attack_days = sorted(redteam_quick['day'].unique())

        logger.info(f"📅 Attack days detected: {attack_days}")

        # Load first few days with attacks for validation (not all to keep it fast)
        validation_days = attack_days[:3] if len(attack_days) >= 3 else attack_days
        logger.info(f"🔍 Validating with days: {validation_days}")

        log_memory_usage("before_loading")
        loader = LANLLoader(data_dir)
        auth_df, redteam_df = loader.load_sample(days=validation_days)
        log_memory_usage("after_loading")

        # Clean up redteam_quick immediately
        aggressive_cleanup(redteam_quick)

        # Limit for validation
        if len(auth_df) > 50000:
            auth_df = auth_df.sample(n=50000, random_state=42)
            logger.info(f"✅ Sampled {len(auth_df):,} events for validation")

        # Check data sufficiency
        data_ok, data_message = check_sufficient_data(auth_df, redteam_df)
        if not data_ok:
            logger.warning(f"⚠️ Data sufficiency issues: {data_message}")

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
            return 0
        else:
            logger.error("❌ VALIDATION FAILED - Fix issues above")
            return 1

        # ✅ ADDED: Cleanup
        log_memory_usage("before_cleanup")
        aggressive_cleanup(auth_df, redteam_df)
        log_memory_usage("after_cleanup")

    except Exception as e:
        logger.error(f"\n❌ Phase 0 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
