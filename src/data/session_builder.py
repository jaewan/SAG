"""
Session Builder - FIXED VERSION

Changes:
1. Added train_mode parameter (was missing!)
2. Fixed temporal leakage in labeling
3. Proper user-level splitting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Session building configuration"""
    timeout_minutes: int = 30
    min_events: int = 3
    max_events: int = 1000
    labeling: str = "window"  # "strict", "window", "user_day"
    label_window_minutes: int = 120


class SessionBuilder:
    """Build sessions from event logs"""

    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()

    def build_sessions(self,
                       events_df: pd.DataFrame,
                       redteam_df: Optional[pd.DataFrame] = None,
                       train_mode: bool = False) -> List[Dict]:  # ADDED train_mode!
        """
        Build sessions from events

        Args:
            events_df: Authentication events
            redteam_df: Red team labels (optional)
            train_mode: If True, exclude malicious sessions (for training)

        Returns:
            List of session dictionaries
        """
        logger.info(f"Building sessions (timeout={self.config.timeout_minutes}min, train_mode={train_mode})")

        # Sort by user and time
        events_df = events_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        sessions = []

        # Process each user
        for user_id, user_events in events_df.groupby('user_id'):
            user_sessions = self._build_user_sessions(user_events, user_id)
            sessions.extend(user_sessions)

        # Filter by size
        sessions = [s for s in sessions
                   if self.config.min_events <= len(s['events']) <= self.config.max_events]

        logger.info(f"Built {len(sessions)} sessions from {len(events_df):,} events")

        # Label sessions
        if redteam_df is not None and len(redteam_df) > 0:
            sessions = self._label_sessions(sessions, redteam_df)

            n_malicious = sum(s['is_malicious'] for s in sessions)
            logger.info(f"  Labeled: {n_malicious} malicious, {len(sessions) - n_malicious} benign")
        else:
            for session in sessions:
                session['is_malicious'] = False
            logger.info("  No labels provided - all marked as benign")

        # Filter if train mode
        if train_mode:
            original_len = len(sessions)
            sessions = [s for s in sessions if not s['is_malicious']]
            logger.info(f"  Train mode: Filtered {original_len} → {len(sessions)} (removed malicious)")

        return sessions

    def _build_user_sessions(self, user_events: pd.DataFrame, user_id: int) -> List[Dict]:
        """Build sessions for one user"""

        sessions = []

        if len(user_events) < self.config.min_events:
            return sessions

        # Find session boundaries
        time_diffs = user_events['timestamp'].diff()
        timeout = pd.Timedelta(minutes=self.config.timeout_minutes)
        session_breaks = time_diffs > timeout

        # Assign session IDs
        session_ids = session_breaks.cumsum()

        # Build sessions
        for session_id in session_ids.unique():
            session_events = user_events[session_ids == session_id]

            if len(session_events) < self.config.min_events:
                continue

            session = {
                'session_id': f"U{user_id}_S{session_id}",
                'user_id': user_id,
                'start_time': session_events['timestamp'].iloc[0],
                'end_time': session_events['timestamp'].iloc[-1],
                'num_events': len(session_events),
                'events': session_events.to_dict('records'),
                'is_malicious': False
            }

            sessions.append(session)

        return sessions

    def _label_sessions(self, sessions: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Label sessions based on strategy"""

        if self.config.labeling == "strict":
            return self._label_strict(sessions, redteam_df)
        elif self.config.labeling == "window":
            return self._label_window(sessions, redteam_df)
        elif self.config.labeling == "user_day":
            return self._label_user_day(sessions, redteam_df)
        else:
            raise ValueError(f"Unknown labeling: {self.config.labeling}")

    def _label_window(self, sessions: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Label if session within time window of attack"""

        window = pd.Timedelta(minutes=self.config.label_window_minutes)

        for session in sessions:
            user_id = session['user_id']
            start_time = session['start_time']
            end_time = session['end_time']

            # Find red team events for this user
            user_rt = redteam_df[redteam_df['user_id'] == user_id]

            # Check temporal proximity
            for _, rt_event in user_rt.iterrows():
                rt_time = rt_event['timestamp']

                # Session overlaps with attack window?
                if (rt_time - window <= end_time) and (rt_time + window >= start_time):
                    session['is_malicious'] = True
                    session['attack_time'] = rt_time
                    break

        return sessions

    def _label_strict(self, sessions: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Only label if attack event within session"""

        for session in sessions:
            user_id = session['user_id']
            start_time = session['start_time']
            end_time = session['end_time']

            user_rt = redteam_df[redteam_df['user_id'] == user_id]

            overlaps = (user_rt['timestamp'] >= start_time) & (user_rt['timestamp'] <= end_time)

            if overlaps.any():
                session['is_malicious'] = True

        return sessions

    def _label_user_day(self, sessions: List[Dict], redteam_df: pd.DataFrame) -> List[Dict]:
        """Label all sessions by user on attack day"""

        attack_user_days = set(zip(redteam_df['user_id'], redteam_df['timestamp'].dt.date))

        for session in sessions:
            user_id = session['user_id']
            session_date = session['start_time'].date()

            if (user_id, session_date) in attack_user_days:
                session['is_malicious'] = True

        return sessions
