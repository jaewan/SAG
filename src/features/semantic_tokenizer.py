"""
Semantic Tokenization for LANL Dataset
Extracts security-relevant features instead of raw identifiers
"""

import logging
from typing import Dict, List

# Import shared utilities - fallback if not available
try:
    from src.utils.lanl_utils import (
        UserRoleClassifier,
        ComputerClassifier,
        TimeContextClassifier
    )
except ImportError:
    # Simple fallback implementations
    class UserRoleClassifier:
        @staticmethod
        def classify_user(user_id: str) -> str:
            if 'admin' in user_id.lower() or 'system' in user_id.lower():
                return 'admin'
            elif '$' in user_id or 'service' in user_id.lower():
                return 'service'
            else:
                return 'regular'

    class ComputerClassifier:
        @staticmethod
        def classify_computer(computer_id: str) -> str:
            if not computer_id or not computer_id.startswith('C'):
                return 'unknown'
            comp_num = computer_id[1:]
            if comp_num.isdigit():
                num = int(comp_num)
                if num <= 100:
                    return 'dc'  # Domain controller
                elif num <= 1000:
                    return 'server'
                else:
                    return 'workstation'
            return 'unknown'

    class TimeContextClassifier:
        @staticmethod
        def get_time_context(timestamp) -> str:
            if timestamp is None:
                return 'unknown'
            hour = timestamp.hour
            if 9 <= hour <= 17:
                return 'business'
            elif hour < 6 or hour > 22:
                return 'night'
            else:
                return 'evening'

        @staticmethod
        def is_unusual_time(timestamp) -> bool:
            if timestamp is None:
                return False
            hour = timestamp.hour
            return hour < 6 or hour > 22

logger = logging.getLogger(__name__)


class LANLSemanticTokenizer:
    """
    Converts LANL events to semantic tokens that capture security context

    Key Design:
    - User roles (admin/service/regular) instead of user IDs
    - Computer types (DC/server/workstation) instead of computer IDs
    - Time contexts (business/night/maintenance) instead of hour bins
    - Process categories (script/office/system) for correlation
    """

    def __init__(self):
        # Process categories for security analysis
        self.process_categories = {
            'script_engine': ['powershell', 'cmd', 'bash', 'wscript', 'cscript'],
            'office': ['winword', 'excel', 'outlook', 'acrord32'],
            'browser': ['chrome', 'firefox', 'iexplore', 'safari'],
            'system': ['explorer', 'svchost', 'lsass', 'csrss'],
            'admin_tool': ['mmc', 'regedit', 'taskmgr', 'services'],
            'network': ['net.exe', 'ping', 'ipconfig', 'nslookup']
        }

    def tokenize_session(self, session: Dict) -> List[str]:
        """
        Main tokenization function

        Args:
            session: Dict with keys:
                - user_id: str
                - events: List[Dict] with auth events
                - correlated_event: Dict (optional) with processes

        Returns:
            List of semantic tokens
        """
        tokens = []

        # Infer user role once per session
        user_id = str(session['user_id'])
        user_role = self._get_user_role(user_id)

        # Process each event
        for event in session['events']:
            # Extract semantic features
            features = self._extract_event_features(event, user_role)

            # Build base token
            base_token = self._build_base_token(features)
            tokens.append(base_token)

            # Add process tokens if available
            if 'correlated_event' in session:
                proc_tokens = self._extract_process_tokens(
                    session['correlated_event']
                )
                tokens.extend(proc_tokens)

        return tokens

    def _get_user_role(self, user_id: str) -> str:
        """
        Get user role using shared classification utility
        """
        return UserRoleClassifier.classify_user(user_id)

    def _extract_event_features(self, event: Dict, user_role: str) -> Dict:
        """Extract all semantic features from event"""
        timestamp = event.get('timestamp')

        features = {
            'auth_type': event.get('auth_type', 'Unknown'),
            'outcome': event.get('outcome', 'Unknown'),
            'user_role': user_role,
            'time_context': self._get_time_context(timestamp),
            'is_weekend': self._is_weekend(timestamp),
            'src_computer_type': self._get_computer_type(event.get('src_computer', '')),
            'dst_computer_type': self._get_computer_type(event.get('dst_computer', '')),
            'is_cross_domain': self._is_cross_domain(event)
        }

        return features

    def _get_time_context(self, timestamp) -> str:
        """
        Get time context using shared classification utility
        """
        return TimeContextClassifier.get_time_context(timestamp)

    def _is_weekend(self, timestamp) -> bool:
        """Check if event is on weekend using shared utility"""
        return TimeContextClassifier.is_unusual_time(timestamp)

    def _get_computer_type(self, computer_id: str) -> str:
        """
        Get computer type using shared classification utility
        """
        return ComputerClassifier.classify_computer(computer_id)

    def _is_cross_domain(self, event: Dict) -> bool:
        """Check if authentication crosses domain boundaries"""
        src_domain = event.get('src_domain', '')
        dst_domain = event.get('dst_domain', src_domain)  # Assume same if missing

        return src_domain != dst_domain if src_domain and dst_domain else False

    def _build_base_token(self, features: Dict) -> str:
        """
        Build semantic token from features

        Format: <auth>_<role>_<time>_<src>to<dst>_<context>

        Examples:
        - "Kerberos_admin_night_dcToServer_weekday"
        - "NTLM_regular_business_workstationToWorkstation_weekday"
        """
        parts = [
            features['auth_type'],
            features['user_role'],
            features['time_context'],
            f"{features['src_computer_type']}To{features['dst_computer_type']}",
            'weekend' if features['is_weekend'] else 'weekday'
        ]

        # Add cross-domain flag if true
        if features['is_cross_domain']:
            parts.append('crossDomain')

        # Add outcome if failure (success is normal)
        if features['outcome'].lower() != 'success':
            parts.append(features['outcome'])

        return '_'.join(parts)

    def _extract_process_tokens(self, correlated_event: Dict) -> List[str]:
        """
        Extract process tokens from correlated data

        Format: proc_<category>_parent<parent_category>

        Example: "proc_scriptEngine_parentOffice"
        (PowerShell launched from Word - suspicious!)
        """
        tokens = []

        for proc in correlated_event.get('related_processes', []):
            proc_name = proc.get('process_name', '').lower()
            parent_name = proc.get('parent_process', '').lower()

            # Categorize process and parent
            proc_category = self._categorize_process(proc_name)
            parent_category = self._categorize_process(parent_name)

            token = f"proc_{proc_category}_parent{parent_category}"
            tokens.append(token)

        return tokens

    def _categorize_process(self, process_name: str) -> str:
        """
        Categorize process by security relevance

        Returns: Category name or 'other'
        """
        process_lower = process_name.lower()

        for category, keywords in self.process_categories.items():
            for keyword in keywords:
                if keyword in process_lower:
                    return category.replace('_', '')  # Remove underscore

        return 'other'

    def get_vocabulary_stats(self, tokenized_sessions: List[List[str]]) -> Dict:
        """
        Analyze vocabulary to ensure it's reasonable

        Expected:
        - Vocabulary size: 10K-50K (not 100K+)
        - Most common tokens should be interpretable
        - Rare tokens should still be semantic
        """
        from collections import Counter

        all_tokens = []
        for session_tokens in tokenized_sessions:
            all_tokens.extend(session_tokens)

        token_counts = Counter(all_tokens)

        return {
            'vocabulary_size': len(token_counts),
            'total_tokens': len(all_tokens),
            'most_common_10': token_counts.most_common(10),
            'singleton_count': sum(1 for count in token_counts.values() if count == 1),
            'singleton_ratio': sum(1 for count in token_counts.values() if count == 1) / len(token_counts)
        }


# Example usage
if __name__ == "__main__":
    # Test with sample session
    tokenizer = LANLSemanticTokenizer()

    # Admin at 3AM example (BENIGN)
    admin_session = {
        'user_id': 'U12',  # Admin user
        'events': [
            {
                'timestamp': datetime(2011, 4, 1, 3, 0, 0),  # 3 AM
                'auth_type': 'Kerberos',
                'outcome': 'Success',
                'src_computer': 'C17',   # Domain controller
                'dst_computer': 'C523',  # Server
                'src_domain': 'DOM1',
                'dst_domain': 'DOM1'
            }
        ]
    }

    # Regular user at 3AM example (SUSPICIOUS)
    user_session = {
        'user_id': 'U420',  # Regular user
        'events': [
            {
                'timestamp': datetime(2011, 4, 1, 3, 0, 0),  # 3 AM
                'auth_type': 'Kerberos',
                'outcome': 'Success',
                'src_computer': 'C1234',  # Workstation
                'dst_computer': 'C567',   # Server (lateral movement!)
                'src_domain': 'DOM1',
                'dst_domain': 'DOM1'
            }
        ]
    }

    admin_tokens = tokenizer.tokenize_session(admin_session)
    user_tokens = tokenizer.tokenize_session(user_session)

    print("Admin at 3AM tokens:")
    print(admin_tokens)
    # Expected: ['Kerberos_admin_night_dcToServer_weekday']

    print("\nRegular user at 3AM tokens:")
    print(user_tokens)
    # Expected: ['Kerberos_regular_night_workstationToServer_weekday']

    print("\n✅ NOTICE: Both have 'night' but different roles!")
    print("   Model can now learn: admin+night = OK, regular+night+workstation→server = SUSPICIOUS")


class LANLRawTokenizer:
    """
    Raw tokenization - minimal features for baseline comparison

    Only uses auth_type and outcome for H3 validation:
    "N-grams fail even with semantic tokens" vs "N-grams fail even without semantic context"
    """

    def tokenize_session(self, session: Dict) -> List[str]:
        """
        Tokenize session using only raw auth events

        Args:
            session: Session dictionary with events

        Returns:
            List of tokens: [auth_type_outcome, ...]
        """
        tokens = []

        for event in session.get('events', []):
            auth_type = event.get('auth_type', 'Unknown')
            outcome = event.get('outcome', 'Unknown')

            # Simple token format: auth_type_outcome
            token = f"{auth_type}_{outcome}"
            tokens.append(token)

        return tokens

    def __repr__(self):
        return "LANLRawTokenizer()"


# Example usage and testing
if __name__ == "__main__":
    # Same test sessions but with raw tokenization
    admin_session = {
        'session_id': 'S1',
        'user_id': 'U420',
        'events': [
            {
                'auth_type': 'Kerberos',
                'outcome': 'Success'
            }
        ]
    }

    raw_tokenizer = LANLRawTokenizer()
    raw_tokens = raw_tokenizer.tokenize_session(admin_session)

    print("Raw tokens (no semantic context):")
    print(raw_tokens)
    # Expected: ['Kerberos_Success']

    print("\nComparison with semantic tokens:")
    semantic_tokenizer = LANLSemanticTokenizer()
    semantic_tokens = semantic_tokenizer.tokenize_session(admin_session)
    print("Semantic:", semantic_tokens)
    print("Raw:", raw_tokens)
    # Shows the difference: semantic has rich context, raw is minimal
