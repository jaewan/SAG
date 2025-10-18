"""
Correlation Quality Validation for LANL Multi-Source Data

This module validates that auth-proc-flow correlation is working correctly.
Low correlation rates are EXPECTED for LANL (processes are sparse).
"""

import logging
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


class CorrelationValidator:
    """Validates multi-source event correlation quality"""

    # LANL-specific thresholds (based on dataset characteristics)
    EXPECTED_RATES = {
        'with_processes': 0.01,      # 1% is normal (processes logged sparsely)
        'with_flows': 0.05,          # 5% is reasonable
        'with_dns': 0.03,            # 3% is reasonable
        'with_any_context': 0.08     # 8% overall is sufficient for Phase 1
    }

    def validate(self, correlated_events: List[Dict],
                 fail_on_poor: bool = False) -> Dict:
        """
        Validate correlation quality

        Args:
            correlated_events: List of correlated event dicts
            fail_on_poor: If True, raise exception on poor correlation

        Returns:
            Quality report dict
        """
        logger.info("="*80)
        logger.info("CORRELATION QUALITY VALIDATION")
        logger.info("="*80)

        if not correlated_events:
            logger.error("❌ No correlated events provided!")
            return {'status': 'empty', 'correlation_rate': 0.0}

        # Calculate rates
        total = len(correlated_events)
        with_processes = sum(1 for e in correlated_events
                           if len(e.get('related_processes', [])) > 0)
        with_flows = sum(1 for e in correlated_events
                        if len(e.get('related_flows', [])) > 0)
        with_dns = sum(1 for e in correlated_events
                      if len(e.get('related_dns', [])) > 0)
        with_any = sum(1 for e in correlated_events
                      if (len(e.get('related_processes', [])) > 0 or
                          len(e.get('related_flows', [])) > 0 or
                          len(e.get('related_dns', [])) > 0))

        report = {
            'total_events': total,
            'with_processes': with_processes,
            'with_flows': with_flows,
            'with_dns': with_dns,
            'with_any_context': with_any,
            'process_rate': with_processes / total,
            'flow_rate': with_flows / total,
            'dns_rate': with_dns / total,
            'correlation_rate': with_any / total
        }

        # Print report
        self._print_report(report)

        # Sample validation
        self._print_samples(correlated_events[:5])

        # Determine status
        status = self._determine_status(report)
        report['status'] = status

        # Check against thresholds
        self._check_thresholds(report, fail_on_poor)

        return report

    def _print_report(self, report: Dict):
        """Print formatted report"""
        logger.info(f"\n📊 Correlation Statistics:")
        logger.info(f"   Total events: {report['total_events']:,}")
        logger.info(f"   With processes: {report['with_processes']:,} ({report['process_rate']*100:.1f}%)")
        logger.info(f"   With flows: {report['with_flows']:,} ({report['flow_rate']*100:.1f}%)")
        logger.info(f"   With DNS: {report['with_dns']:,} ({report['dns_rate']*100:.1f}%)")
        logger.info(f"   With ANY context: {report['with_any_context']:,} ({report['correlation_rate']*100:.1f}%)")

    def _print_samples(self, samples: List[Dict]):
        """Print sample correlated events for inspection"""
        logger.info(f"\n🔍 Sample Correlated Events:")

        for i, event in enumerate(samples, 1):
            auth = event['auth_event']
            logger.info(f"\n  Event {i}:")
            logger.info(f"    Time: {auth.get('timestamp')}")
            logger.info(f"    User: {auth.get('user_id')}")
            logger.info(f"    Auth: {auth.get('auth_type')} → {auth.get('outcome')}")
            logger.info(f"    Related processes: {len(event.get('related_processes', []))}")
            logger.info(f"    Related flows: {len(event.get('related_flows', []))}")
            logger.info(f"    Related DNS: {len(event.get('related_dns', []))}")

            # Show process details if available
            if event.get('related_processes'):
                procs = event['related_processes'][:3]  # First 3
                logger.info(f"    Sample processes:")
                for proc in procs:
                    logger.info(f"      → {proc.get('process_name')} (parent: {proc.get('parent_process', 'N/A')})")

    def _determine_status(self, report: Dict) -> str:
        """Determine overall correlation status"""
        rate = report['correlation_rate']

        if rate >= self.EXPECTED_RATES['with_any_context']:
            return 'good'
        elif rate >= self.EXPECTED_RATES['with_any_context'] * 0.5:
            return 'marginal'
        else:
            return 'poor'

    def _check_thresholds(self, report: Dict, fail_on_poor: bool):
        """Check against expected thresholds"""
        logger.info(f"\n📋 Threshold Comparison:")
        logger.info(f"   Expected minimum: {self.EXPECTED_RATES['with_any_context']*100:.1f}%")
        logger.info(f"   Actual: {report['correlation_rate']*100:.1f}%")

        if report['status'] == 'good':
            logger.info("   ✅ Status: GOOD - Correlation working well")
        elif report['status'] == 'marginal':
            logger.warning("   ⚠️ Status: MARGINAL - Lower than expected but acceptable")
            logger.warning("      Possible causes:")
            logger.warning("      - Processes are naturally sparse in LANL")
            logger.warning("      - Correlation window may need tuning")
            logger.warning("      - User/computer ID format mismatches")
        else:  # poor
            logger.error("   ❌ Status: POOR - Correlation may be broken")
            logger.error("      This is a CRITICAL issue! Possible causes:")
            logger.error("      1. Timestamp formats don't align across files")
            logger.error("      2. User/Computer ID extraction failing")
            logger.error("      3. Correlation window too narrow (try ±15 min)")
            logger.error("      4. Process/flow data not from same time period")

            if fail_on_poor:
                raise RuntimeError(
                    f"Correlation quality too low ({report['correlation_rate']*100:.1f}%). "
                    f"Expected >= {self.EXPECTED_RATES['with_any_context']*100:.1f}%"
                )

    def validate_malicious_correlation(self, correlated_events: List[Dict]) -> Dict:
        """
        Specifically validate that malicious events have good correlation
        (This is critical - attacks should trigger multiple log sources)
        """
        malicious = [e for e in correlated_events if e.get('is_malicious', False)]

        if not malicious:
            logger.warning("⚠️ No malicious events to validate correlation")
            return {'status': 'no_data'}

        logger.info(f"\n🔍 Malicious Event Correlation:")
        logger.info(f"   Total malicious: {len(malicious)}")

        with_context = sum(1 for e in malicious
                          if (len(e.get('related_processes', [])) > 0 or
                              len(e.get('related_flows', [])) > 0))

        mal_correlation_rate = with_context / len(malicious)
        logger.info(f"   With context: {with_context} ({mal_correlation_rate*100:.1f}%)")

        if mal_correlation_rate < 0.1:  # Less than 10% of attacks have context
            logger.warning("⚠️ LOW: Most attacks lack correlated context")
            logger.warning("   This is OK for Phase 1 (auth-only baseline)")
            logger.warning("   But important for SAG (needs rich context)")

        return {
            'malicious_count': len(malicious),
            'with_context': with_context,
            'correlation_rate': mal_correlation_rate,
            'status': 'good' if mal_correlation_rate > 0.1 else 'poor'
        }
