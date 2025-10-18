"""
Shared Utilities for LANL Dataset

Centralized classification logic to eliminate duplication
"""

import re
from typing import Dict, Any
from datetime import datetime


class UserRoleClassifier:
    """Centralized user role classification for LANL dataset with caching"""

    ADMIN_PATTERNS = [
        r'^U[0-9]{1,2}$',      # U1-U99 are admins (LANL convention)
        r'ADMIN',
        r'SYSTEM',
        r'Administrator',
        r'ROOT'
    ]

    SERVICE_PATTERNS = [
        r'^U5[0-9]{2}$',       # U500-U599 are service accounts
        r'ANONYMOUS',
        r'SERVICE',
        r'\$$',                # Computer accounts end with $
        r'NETWORK SERVICE',
        r'LOCAL SERVICE'
    ]

    # ✅ ADDED: Cache for expensive regex operations
    _user_cache: Dict[str, str] = {}

    @staticmethod
    def classify_user(user_id: str) -> str:
        """
        Classify user role with caching for performance

        Args:
            user_id: User identifier

        Returns:
            'admin', 'service', or 'regular'
        """
        # ✅ ADDED: Check cache first for expensive regex operations
        cache_key = str(user_id).upper()
        if cache_key in UserRoleClassifier._user_cache:
            return UserRoleClassifier._user_cache[cache_key]

        user_upper = str(user_id).upper()

        # Check admin patterns
        for pattern in UserRoleClassifier.ADMIN_PATTERNS:
            if re.search(pattern, user_upper):
                UserRoleClassifier._user_cache[cache_key] = 'admin'
                return 'admin'

        # Check service patterns
        for pattern in UserRoleClassifier.SERVICE_PATTERNS:
            if re.search(pattern, user_upper):
                UserRoleClassifier._user_cache[cache_key] = 'service'
                return 'service'

        # Default: regular user
        UserRoleClassifier._user_cache[cache_key] = 'regular'
        return 'regular'

    @staticmethod
    def is_admin(user_id: str) -> bool:
        """Check if user is admin"""
        return UserRoleClassifier.classify_user(user_id) == 'admin'

    @staticmethod
    def is_service_account(user_id: str) -> bool:
        """Check if user is service account"""
        return UserRoleClassifier.classify_user(user_id) == 'service'


class ComputerClassifier:
    """Centralized computer type classification for LANL dataset with caching"""

    # ✅ ADDED: Cache for expensive regex operations
    _computer_cache: Dict[str, str] = {}

    @staticmethod
    def classify_computer(computer_id: str) -> str:
        """
        Classify computer type based on LANL ID ranges with caching

        LANL Conventions:
        - C1-C499: Domain controllers and critical servers
        - C500-C999: Application servers
        - C1000-C17684: Workstations

        Args:
            computer_id: Computer identifier (e.g., 'C1234')

        Returns:
            'dc', 'server', 'workstation', or 'unknown'
        """
        # ✅ ADDED: Check cache first for expensive regex operations
        cache_key = str(computer_id)
        if cache_key in ComputerClassifier._computer_cache:
            return ComputerClassifier._computer_cache[cache_key]

        match = re.search(r'C(\d+)', str(computer_id))
        if not match:
            ComputerClassifier._computer_cache[cache_key] = 'unknown'
            return 'unknown'

        comp_num = int(match.group(1))

        result = None
        if comp_num < 500:
            result = 'dc'  # Domain controller
        elif comp_num < 1000:
            result = 'server'
        else:
            result = 'workstation'

        ComputerClassifier._computer_cache[cache_key] = result
        return result

    @staticmethod
    def is_critical(computer_id: str) -> bool:
        """Check if computer is critical infrastructure"""
        comp_type = ComputerClassifier.classify_computer(computer_id)
        return comp_type in ['dc', 'server']


class TimeContextClassifier:
    """Centralized time context classification"""

    @staticmethod
    def get_time_context(timestamp) -> str:
        """
        Classify time into semantic categories

        Args:
            timestamp: pandas Timestamp or datetime

        Returns:
            'business', 'evening', 'night', 'early', or 'unknown'
        """
        if timestamp is None:
            return 'unknown'

        hour = timestamp.hour

        if 9 <= hour < 17:
            return 'business'  # 9 AM - 5 PM
        elif 17 <= hour < 22:
            return 'evening'   # 5 PM - 10 PM
        elif 22 <= hour or hour < 6:
            return 'night'     # 10 PM - 6 AM (KEY for 3AM example!)
        else:
            return 'early'     # 6 AM - 9 AM

    @staticmethod
    def is_business_hours(timestamp) -> bool:
        """Check if timestamp is during business hours"""
        if timestamp is None:
            return False
        return (9 <= timestamp.hour < 17) and (timestamp.weekday() < 5)

    @staticmethod
    def is_maintenance_window(timestamp) -> bool:
        """
        Check if timestamp is during scheduled maintenance

        Default maintenance: Sunday 2-4 AM
        """
        if timestamp is None:
            return False

        # Sunday = 6 in Python weekday()
        return (timestamp.weekday() == 6 and
                2 <= timestamp.hour < 4)

    @staticmethod
    def is_unusual_time(timestamp) -> bool:
        """Check if time is unusual (night/weekend)"""
        if timestamp is None:
            return False

        is_night = timestamp.hour < 6 or timestamp.hour >= 22
        is_weekend = timestamp.weekday() >= 5

        return is_night or is_weekend


# Convenience functions for backward compatibility
def classify_user_role(user_id: str) -> str:
    """Classify user role (convenience function)"""
    return UserRoleClassifier.classify_user(user_id)


def classify_computer_type(computer_id: str) -> str:
    """Classify computer type (convenience function)"""
    return ComputerClassifier.classify_computer(computer_id)


def get_time_context(timestamp) -> str:
    """Get time context (convenience function)"""
    return TimeContextClassifier.get_time_context(timestamp)
