"""
Memory monitoring utilities for production use
"""

import psutil
import logging
import time
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during execution"""

    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}

    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        logger.info("🔍 Memory monitoring started")

    def log_usage(self, stage: str):
        """Log memory usage at current stage"""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / 1024 / 1024 / 1024
            logger.info(f"💾 Memory at {stage}: {mem_gb:.2f} GB")
            self.checkpoints[stage] = mem_gb
        except Exception as e:
            logger.warning(f"Could not log memory at {stage}: {e}")

    def print_summary(self):
        """Print memory usage summary"""
        if not self.checkpoints:
            logger.info("📊 No memory checkpoints recorded")
            return

        logger.info("📊 Memory Usage Summary:")
        for stage, mem_gb in self.checkpoints.items():
            logger.info(f"  {stage}: {mem_gb:.2f} GB")

        total_time = time.time() - self.start_time
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds")


def memory_safe(max_memory_gb: float = 8.0):
    """
    Decorator to check memory before running function and abort if insufficient
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                process = psutil.Process()
                used_gb = process.memory_info().rss / (1024**3)
                available_gb = psutil.virtual_memory().available / (1024**3)

                # Check if we're approaching the limit
                if used_gb > max_memory_gb * 0.9:
                    logger.error(f"❌ Memory limit exceeded: {used_gb:.1f}GB > {max_memory_gb*0.9:.1f}GB")
                    logger.error(f"   Available system memory: {available_gb:.1f}GB")
                    logger.error("   Solutions:")
                    logger.error("   1. Close other applications")
                    logger.error("   2. Reduce data size")
                    logger.error("   3. Use smaller batch sizes")
                    raise MemoryError(f"Memory limit exceeded: {used_gb:.1f}GB")

                logger.info(f"✅ Memory check passed: {used_gb:.1f}GB used, {available_gb:.1f}GB available")
                return func(*args, **kwargs)
            except MemoryError:
                raise  # Re-raise MemoryError
            except Exception as e:
                logger.error(f"Memory check failed: {e}")
                # Continue anyway for robustness
                return func(*args, **kwargs)
        return wrapper
    return decorator


def check_memory_or_abort(operation_name: str, min_gb: float = 2.0) -> bool:
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


def get_memory_info() -> dict:
    """Get comprehensive memory information"""
    try:
        vm = psutil.virtual_memory()
        process = psutil.Process()

        return {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'process_mb': process.memory_info().rss / 1024 / 1024,
            'process_percent': process.memory_percent()
        }
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return {}
