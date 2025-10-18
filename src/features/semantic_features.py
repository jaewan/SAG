"""
Fixed Semantic Features for Phase 1.5

CRITICAL FIX: Handles missing parent_process field in LANL dataset
Uses temporal sequences and co-occurrence as proxy for process chains
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_semantic_features_comprehensive(rare_events: List[Dict]) -> pd.DataFrame:
    """
    Extract comprehensive semantic features for rare events

    Args:
        rare_events: List of correlated event dictionaries

    Returns:
        DataFrame with semantic features
    """
    logger.info(f"🎯 Extracting comprehensive semantic features for {len(rare_events)} events...")

    features = []

    for event in rare_events:
        feat = {}

        # Auth features
        auth_event = event.get('auth_event', {})
        feat['user_id'] = auth_event.get('user_id', '')
        feat['src_computer'] = auth_event.get('src_computer', '')
        feat['dst_computer'] = auth_event.get('dst_computer', '')
        feat['auth_type'] = auth_event.get('auth_type', '')
        feat['outcome'] = auth_event.get('outcome', '')
        feat['timestamp'] = auth_event.get('timestamp')

        # Semantic user features
        feat['user_is_admin'] = 'admin' in str(feat['user_id']).lower() or 'system' in str(feat['user_id']).lower()
        feat['user_is_service'] = '$' in str(feat['user_id']) or 'service' in str(feat['user_id']).lower()

        # Temporal features
        if feat['timestamp']:
            feat['hour'] = feat['timestamp'].hour
            feat['day_of_week'] = feat['timestamp'].dayofweek
            feat['is_business_hours'] = 9 <= feat['hour'] <= 17 and feat['day_of_week'] < 5
            feat['is_unusual_time'] = feat['hour'] < 6 or feat['hour'] > 22
            feat['is_weekend'] = feat['day_of_week'] >= 5
        else:
            feat['hour'] = -1
            feat['is_business_hours'] = False
            feat['is_unusual_time'] = False

        # Process features (FIXED - no parent_process in LANL)
        processes = event.get('related_processes', [])
        process_features = extract_process_features_fixed(processes)
        feat.update(process_features)

        # Network features
        flows = event.get('related_flows', [])
        network_features = extract_network_features(flows)
        feat.update(network_features)

        # DNS features
        dns_queries = event.get('related_dns', [])
        dns_features = extract_dns_features(dns_queries)
        feat.update(dns_features)

        # Overall suspicion scores
        feat['has_process'] = len(processes) > 0
        feat['has_network'] = len(flows) > 0
        feat['has_dns'] = len(dns_queries) > 0

        # Multi-source suspicion score
        feat['overall_suspicion'] = max(
            feat.get('process_suspicion', 0),
            feat.get('network_suspicion', 0),
            feat.get('dns_suspicion', 0)
        )

        features.append(feat)

    df = pd.DataFrame(features)
    logger.info(f"✅ Extracted {len(df)} events with {len(df.columns)} features")

    return df


def extract_process_features_fixed(processes: List[Dict]) -> Dict:
    """
    Extract process features without relying on parent_process field

    LANL proc.txt format: time, user@domain, computer@domain, process_name, start_time
    NO parent_process field! Must use temporal sequences and co-occurrence.
    """
    if not processes:
        return {
            'has_process': False,
            'process_count': 0,
            'process_diversity': 0,
            'process_is_shell': False,
            'process_is_office': False,
            'process_is_browser': False,
            'process_suspicion': 0.0,
            'suspicious_sequence': 0.0,
            'suspicious_cooccurrence': 0.0
        }

    # Basic process info
    proc_names = [p.get('process_name', '').lower() for p in processes]
    unique_procs = set(proc_names)

    features = {
        'has_process': True,
        'process_count': len(processes),
        'process_diversity': len(unique_procs) / len(processes) if processes else 0,
        'process_is_shell': any('powershell' in p or 'cmd' in p for p in proc_names),
        'process_is_office': any(o in p for o in ['winword', 'excel', 'outlook'] for p in proc_names),
        'process_is_browser': any(b in p for b in ['chrome', 'firefox', 'iexplore'] for p in proc_names),
    }

    # Temporal sequence scoring (FIXED - no parent_process)
    features['suspicious_sequence'] = score_process_sequence_temporal(processes)

    # Co-occurrence scoring
    features['suspicious_cooccurrence'] = score_process_cooccurrence(processes)

    # Overall process suspicion
    features['process_suspicion'] = max(
        features['suspicious_sequence'],
        features['suspicious_cooccurrence']
    )

    return features


def score_process_sequence_temporal(processes: List[Dict]) -> float:
    """
    Score suspicious temporal sequences without parent_process

    Heuristic: If office app appears before shell within 5 min,
    likely suspicious (even without true parent-child link)
    """
    if len(processes) < 2:
        return 0.0

    # Sort by timestamp
    sorted_procs = sorted(processes, key=lambda p: p.get('timestamp', 0))

    suspicious_score = 0.0

    # Define suspicious temporal sequences
    SUSPICIOUS_SEQUENCES = [
        # (precursor, follower, max_time_gap_sec, score)
        (['winword.exe', 'excel.exe', 'outlook.exe'],
         ['powershell.exe', 'cmd.exe'], 300, 0.8),
        (['chrome.exe', 'firefox.exe', 'iexplore.exe'],
         ['powershell.exe', 'cmd.exe'], 300, 0.6),
        (['explorer.exe'],
         ['net.exe', 'sc.exe', 'at.exe'], 300, 0.7),
    ]

    for i in range(len(sorted_procs) - 1):
        current_proc = sorted_procs[i]['process_name'].lower()
        current_time = sorted_procs[i].get('timestamp', 0)

        for j in range(i+1, len(sorted_procs)):
            next_proc = sorted_procs[j]['process_name'].lower()
            next_time = sorted_procs[j].get('timestamp', current_time)
            time_diff = next_time - current_time

            # Check each suspicious sequence pattern
            for precursors, followers, max_gap, score in SUSPICIOUS_SEQUENCES:
                if time_diff <= max_gap:
                    if any(p in current_proc for p in precursors):
                        if any(f in next_proc for f in followers):
                            suspicious_score = max(suspicious_score, score)

    return min(suspicious_score, 1.0)


def score_process_cooccurrence(processes: List[Dict]) -> float:
    """
    Check if suspicious process combinations occur in same session
    Even without knowing parent-child, co-occurrence is suspicious
    """
    proc_names = [p.get('process_name', '').lower() for p in processes]
    proc_set = set(proc_names)

    SUSPICIOUS_COMBINATIONS = [
        # (set of processes, score, description)
        ({'winword.exe', 'powershell.exe'}, 0.8, 'Office + Shell'),
        ({'outlook.exe', 'powershell.exe'}, 0.8, 'Email + Shell'),
        ({'chrome.exe', 'powershell.exe'}, 0.6, 'Browser + Shell'),
        ({'powershell.exe', 'net.exe'}, 0.7, 'Shell + Network tool'),
        ({'cmd.exe', 'reg.exe'}, 0.6, 'Shell + Registry'),
    ]

    max_score = 0.0
    for combo, score, desc in SUSPICIOUS_COMBINATIONS:
        if combo.issubset(proc_set):
            max_score = max(max_score, score)
            logger.debug(f"Found suspicious combo: {desc}")

    return max_score


def extract_network_features(flows: List[Dict]) -> Dict:
    """Extract network features from correlated flows"""
    if not flows:
        return {
            'has_network': False,
            'network_external': False,
            'high_volume': False,
            'network_suspicion': 0.0
        }

    features = {
        'has_network': True,
        'network_external': False,
        'high_volume': False,
        'network_suspicion': 0.0
    }

    # Analyze flows
    for flow in flows:
        dst_port = flow.get('dst_port', 0)

        # External connections (common attack ports)
        if dst_port in [80, 443, 53, 25, 110, 143, 993, 995]:
            features['network_external'] = True

        # High volume (>1MB)
        byte_count = flow.get('byte_count', 0)
        if byte_count > 1e6:
            features['high_volume'] = True

    # Suspicion score based on external + high volume
    if features['network_external'] and features['high_volume']:
        features['network_suspicion'] = 0.8
    elif features['network_external']:
        features['network_suspicion'] = 0.4
    elif features['high_volume']:
        features['network_suspicion'] = 0.3

    return features


def extract_dns_features(dns_queries: List[Dict]) -> Dict:
    """Extract DNS features from correlated queries"""
    if not dns_queries:
        return {
            'has_dns': False,
            'dns_external': False,
            'dns_suspicion': 0.0
        }

    features = {
        'has_dns': True,
        'dns_external': False,
        'dns_suspicion': 0.0
    }

    # Analyze DNS queries
    for query in dns_queries:
        domain = query.get('domain', '').lower()

        # External domains (not .local, .lan, internal)
        if not any(x in domain for x in ['.local', '.lan', 'internal']):
            features['dns_external'] = True

    # Suspicion based on external DNS
    if features['dns_external']:
        features['dns_suspicion'] = 0.5

    return features


def extract_semantic_features_batch(events: List[Dict]) -> pd.DataFrame:
    """
    Batch extraction of semantic features for efficiency
    """
    all_features = []

    for event in events:
        features = {}

        # Auth features
        auth_event = event.get('auth_event', event)  # Handle both formats
        features['user_id'] = auth_event.get('user_id', '')
        features['timestamp'] = auth_event.get('timestamp')
        features['auth_type'] = auth_event.get('auth_type', '')
        features['outcome'] = auth_event.get('outcome', '')

        # Semantic features
        features['user_is_admin'] = 'admin' in str(features['user_id']).lower()
        features['is_business_hours'] = False
        features['is_maintenance_window'] = False
        features['has_suspicious_processes'] = False
        features['unusual_network_activity'] = False

        if features['timestamp']:
            hour = features['timestamp'].hour
            features['is_business_hours'] = 9 <= hour <= 17
            features['is_maintenance_window'] = 2 <= hour <= 5  # 2-5 AM maintenance

        # Process features (simplified)
        processes = event.get('related_processes', [])
        if processes:
            proc_names = [p.get('process_name', '').lower() for p in processes]
            features['has_suspicious_processes'] = any(
                'powershell' in p or 'cmd' in p for p in proc_names
            )

        # Network features (simplified)
        flows = event.get('related_flows', [])
        if flows:
            features['unusual_network_activity'] = any(
                f.get('dst_port', 0) in [80, 443, 53] for f in flows
            )

        # Overall suspicion
        features['is_malicious'] = event.get('is_malicious', False)

        all_features.append(features)

    return pd.DataFrame(all_features)


def extract_process_features_comprehensive(processes: List[Dict]) -> Dict:
    """
    Comprehensive process feature extraction for ablation study
    """
    return extract_process_features_fixed(processes)
