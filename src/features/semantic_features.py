"""
Semantic Feature Extraction for LANL Dataset
Implements features from the SAG proposal using correlated multi-source data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Critical system paths that should be monitored
CRITICAL_PATHS = {
    'windows': [
        r'C:\Windows\System32',
        r'C:\Windows\SysWOW64',
        r'C:\Program Files',
        r'C:\Program Files (x86)'
    ],
    'linux': [
        '/bin',
        '/sbin',
        '/usr/bin',
        '/usr/sbin',
        '/etc'
    ]
}

# Maintenance windows (example schedule)
MAINTENANCE_WINDOWS = [
    # Weekly maintenance: Sunday 2-4 AM
    {'day': 6, 'start_hour': 2, 'end_hour': 4},
    # Monthly maintenance: First Sunday of month 1-3 AM
    {'day': 6, 'start_hour': 1, 'end_hour': 3, 'day_of_month': 1}
]

# Suspicious process patterns (threat intelligence based)
SUSPICIOUS_PATTERNS = {
    'office_to_shell': [
        ('winword.exe', 'powershell.exe'),
        ('excel.exe', 'powershell.exe'),
        ('outlook.exe', 'powershell.exe'),
        ('winword.exe', 'cmd.exe'),
        ('excel.exe', 'cmd.exe')
    ],
    'browser_to_shell': [
        ('chrome.exe', 'powershell.exe'),
        ('firefox.exe', 'powershell.exe'),
        ('iexplore.exe', 'powershell.exe')
    ],
    'suspicious_processes': [
        'mimikatz.exe',
        'psexec.exe',
        'net.exe',
        'sc.exe'
    ]
}


class SemanticFeatureExtractor:
    """Extract semantic features from correlated LANL events"""

    def __init__(self):
        self.user_history = {}  # Cache for user behavior patterns
        self.host_graph = {}    # Cache for network topology

    def extract_features(self, correlated_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all semantic features for a correlated event

        Args:
            correlated_event: Event with auth, processes, flows, and DNS data

        Returns:
            Dictionary of semantic features
        """
        features = {}

        # Entity-State Features
        features.update(self._extract_entity_state_features(correlated_event))

        # Relational Features
        features.update(self._extract_relational_features(correlated_event))

        # Temporal Features
        features.update(self._extract_temporal_features(correlated_event))

        return features

    def _extract_entity_state_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entity-state features (properties of individual entities)"""
        features = {}
        auth_event = event.get('auth_event', {})

        # User role features
        user_id = auth_event.get('user_id', '')
        features['user_is_admin'] = self._is_admin_user(user_id)
        features['user_login_frequency'] = self._get_user_login_frequency(user_id)

        # Process features
        processes = event.get('related_processes', [])
        if processes:
            features['process_is_signed'] = self._check_process_signatures(processes)
            features['has_suspicious_processes'] = self._check_suspicious_processes(processes)

        return features

    def _extract_relational_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relational features (relationships between entities)"""
        features = {}

        # Process chain features
        processes = event.get('related_processes', [])
        if processes:
            features['parent_child_suspicious'] = self._score_process_chain(processes)
            features['user_process_legitimate'] = self._check_user_process_authorization(
                event['auth_event']['user_id'], processes
            )

        # Network features
        flows = event.get('related_flows', [])
        if flows:
            features['network_direction'] = self._classify_traffic_direction(flows)
            features['external_connections'] = self._count_external_connections(flows)
            features['unusual_network_activity'] = self._detect_unusual_network_activity(flows)

        # File access patterns (if available)
        # Note: LANL doesn't have direct file access logs, but we can infer from processes

        return features

    def _extract_temporal_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features (time-based context)"""
        features = {}
        timestamp = event.get('timestamp')

        if timestamp:
            features['is_business_hours'] = self._is_business_hours(timestamp)
            features['is_maintenance_window'] = self._is_maintenance_window(timestamp)
            features['hour_of_day'] = timestamp.hour
            features['day_of_week'] = timestamp.dayofweek
            features['is_weekend'] = timestamp.dayofweek >= 5

        return features

    def _is_admin_user(self, user_id: str) -> bool:
        """Check if user has admin privileges"""
        # Heuristic: admin users typically have 'admin', 'system', or 'administrator' in name
        admin_indicators = ['admin', 'system', 'administrator', 'root']
        return any(indicator in user_id.lower() for indicator in admin_indicators)

    def _get_user_login_frequency(self, user_id: str) -> float:
        """Get user's login frequency (normalized)"""
        # This would typically be calculated from historical data
        # For now, return a placeholder based on user type
        if self._is_admin_user(user_id):
            return 0.8  # Admins log in more frequently
        else:
            return 0.3  # Regular users log in less frequently

    def _check_process_signatures(self, processes: List[Dict]) -> bool:
        """Check if processes are properly signed"""
        # Placeholder: In real implementation, would check digital signatures
        # For now, assume system processes are signed
        system_processes = {'svchost.exe', 'lsass.exe', 'winlogon.exe', 'explorer.exe'}
        return any(proc['process_name'].lower() in system_processes
                  for proc in processes)

    def _check_suspicious_processes(self, processes: List[Dict]) -> bool:
        """Check for suspicious process names"""
        for proc in processes:
            proc_name = proc.get('process_name', '').lower()
            if any(suspicious in proc_name for suspicious in SUSPICIOUS_PATTERNS['suspicious_processes']):
                return True
        return False

    def _score_process_chain(self, processes: List[Dict]) -> float:
        """Score process parent-child relationships for suspiciousness"""
        if len(processes) < 2:
            return 0.0

        # Sort processes by timestamp
        sorted_processes = sorted(processes, key=lambda p: p['timestamp'])

        suspicious_score = 0.0

        for i in range(len(sorted_processes) - 1):
            parent = sorted_processes[i]['process_name'].lower()
            child = sorted_processes[i + 1]['process_name'].lower()

            # Check for suspicious patterns
            for office_proc, shell_proc in SUSPICIOUS_PATTERNS['office_to_shell']:
                if office_proc.lower() in parent and shell_proc.lower() in child:
                    suspicious_score += 0.8  # High suspicion

            for browser_proc, shell_proc in SUSPICIOUS_PATTERNS['browser_to_shell']:
                if browser_proc.lower() in parent and shell_proc.lower() in child:
                    suspicious_score += 0.6  # Medium suspicion

        return min(suspicious_score, 1.0)

    def _check_user_process_authorization(self, user_id: str, processes: List[Dict]) -> bool:
        """Check if user is authorized to run these processes"""
        if not processes:
            return True

        # Admin users can run more processes
        if self._is_admin_user(user_id):
            return True

        # Regular users should not run system/admin processes
        system_processes = {'cmd.exe', 'powershell.exe', 'net.exe', 'sc.exe'}
        for proc in processes:
            if proc['process_name'].lower() in system_processes:
                return False

        return True

    def _classify_traffic_direction(self, flows: List[Dict]) -> str:
        """Classify network traffic as internal or external"""
        external_indicators = {80, 443, 53}  # Common external ports

        for flow in flows:
            dst_port = flow.get('dst_port', 0)
            if dst_port in external_indicators:
                return 'external'

        return 'internal'

    def _count_external_connections(self, flows: List[Dict]) -> int:
        """Count number of external connections"""
        external_indicators = {80, 443, 53, 25, 110, 995, 993}
        count = 0

        for flow in flows:
            if flow.get('dst_port') in external_indicators:
                count += 1

        return count

    def _detect_unusual_network_activity(self, flows: List[Dict]) -> bool:
        """Detect unusual network patterns"""
        if not flows:
            return False

        # Check for unusual protocols or high packet counts
        for flow in flows:
            if flow.get('packet_count', 0) > 10000:  # Unusual high packet count
                return True
            if flow.get('protocol', 0) not in [6, 17]:  # Not TCP or UDP
                return True

        return False

    def _is_business_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is during business hours"""
        return (9 <= timestamp.hour <= 17) and (timestamp.dayofweek < 5)

    def _is_maintenance_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is during scheduled maintenance"""
        for window in MAINTENANCE_WINDOWS:
            if (timestamp.dayofweek == window['day'] and
                window['start_hour'] <= timestamp.hour < window['end_hour']):
                return True
        return False


def extract_semantic_features_batch(correlated_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract semantic features for a batch of correlated events

    Args:
        correlated_events: List of correlated event dictionaries

    Returns:
        DataFrame with features for all events
    """
    logger.info(f"Extracting semantic features for {len(correlated_events)} events...")

    extractor = SemanticFeatureExtractor()
    features_list = []
    success_count = 0
    default_count = 0

    for i, event in enumerate(correlated_events):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i:,}/{len(correlated_events):,} events...")

        try:
            features = extractor.extract_features(event)

            # Validate that features are meaningful (not all defaults)
            if has_meaningful_features(features):
                success_count += 1
            else:
                default_count += 1
                logger.warning(f"Event {i} has default features - check correlation quality")

            features['event_index'] = i
            features['is_malicious'] = event.get('is_malicious', False)
            features_list.append(features)

        except Exception as e:
            logger.warning(f"Failed to extract features for event {i}: {e}")
            default_count += 1

            # Add default features with error flag
            features_list.append({
                'event_index': i,
                'is_malicious': event.get('is_malicious', False),
                'user_is_admin': False,
                'parent_child_suspicious': 0.0,
                'is_business_hours': True,
                'is_maintenance_window': False,
                'feature_extraction_error': str(e)
            })

    feature_df = pd.DataFrame(features_list)

    # Check if too many features are defaults (indicates correlation problems)
    default_rate = default_count / len(correlated_events) if correlated_events else 0

    logger.info("✅ Feature extraction complete:")
    logger.info(f"   Success rate: {success_count}/{len(feature_df)} ({success_count/len(feature_df)*100:.1f}%)")
    logger.info(f"   Default rate: {default_count}/{len(feature_df)} ({default_rate*100:.1f}%)")

    if default_rate > 0.5:  # More than 50% defaults
        logger.error("❌ High default feature rate - correlation may be failing!")
        logger.error("   This suggests:")
        logger.error("   1. No related processes/flows found for most auth events")
        logger.error("   2. Computer ID formats don't match across files")
        logger.error("   3. Timestamp formats are inconsistent")
        logger.error("   Consider debugging correlation before proceeding")

    elif default_rate > 0.2:  # More than 20% defaults
        logger.warning("⚠️ Moderate default feature rate - check correlation quality")

    else:
        logger.info("✅ Feature extraction quality looks good")

    return feature_df


def has_meaningful_features(features: Dict[str, Any]) -> bool:
    """
    Check if extracted features are meaningful (not all defaults)

    Args:
        features: Dictionary of extracted features

    Returns:
        True if features contain meaningful information
    """
    # Check for non-default values in key features
    meaningful_indicators = [
        features.get('parent_child_suspicious', 0) > 0,
        features.get('user_is_admin', False),
        not features.get('is_business_hours', True),  # Unusual hours is meaningful
        features.get('is_maintenance_window', False),
        features.get('has_suspicious_processes', False),
        features.get('external_connections', 0) > 0,
        features.get('unusual_network_activity', False)
    ]

    return any(meaningful_indicators)
