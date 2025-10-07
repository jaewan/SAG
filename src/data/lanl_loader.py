"""
LANL Dataset Loader - PRODUCTION VERSION
Handles loading and preprocessing of LANL authentication and red team data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import gzip
import warnings

logger = logging.getLogger(__name__)


class LANLValidator:
    """Validate LANL dataset"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.auth_file = self.data_dir / "auth.txt"
        self.redteam_file = self.data_dir / "redteam.txt"

    def _check_timestamps(self) -> Tuple[bool, datetime]:
        """Validate and DETECT start time"""

        if not self.auth_file.exists():
            logger.error(f"❌ Auth file not found: {self.auth_file}")
            return False, datetime(2011, 4, 1, 0, 0, 0)

        # Read CSV - handle both with and without headers
        try:
            df = pd.read_csv(self.auth_file, nrows=10000, header=None)
            # Assume no headers and assign column names based on position
            df.columns = ['time', 'user_id', 'src_comp_id', 'dst_comp_id', 'auth_type', 'outcome']
        except:
            # Try with headers
            df = pd.read_csv(self.auth_file, nrows=10000)
            # Check required columns
            required_cols = ['time', 'user_id', 'src_comp_id', 'dst_comp_id', 'auth_type', 'outcome']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"❌ Missing columns: {missing_cols}")
                return False, datetime(2011, 4, 1, 0, 0, 0)

        # Detect start time
        candidates = [
            datetime(2011, 4, 1, 0, 0, 0),   # Midnight
            datetime(2011, 4, 1, 8, 0, 0),   # 8 AM (docs say this)
        ]

        for start in candidates:
            df['timestamp'] = start + pd.to_timedelta(df['time'], unit='s')

            # Check: Does first event fall on start date?
            first_date = df['timestamp'].iloc[0].date()
            if first_date == start.date():
                logger.info(f"✅ Detected start time: {start}")
                return True, start  # RETURN IT

        logger.warning("⚠️ Could not auto-detect start time, using midnight")
        return True, datetime(2011, 4, 1, 0, 0, 0)

    def validate(self) -> Tuple[bool, datetime]:
        """Run all validations"""
        logger.info("🔍 Validating LANL dataset...")

        # Check files exist
        if not self.auth_file.exists():
            logger.error(f"❌ Auth file missing: {self.auth_file}")
            return False, datetime(2011, 4, 1, 0, 0, 0)

        if not self.redteam_file.exists():
            logger.warning(f"⚠️ Red team file missing: {self.redteam_file}")

        # Check timestamps and get start time
        valid, start_time = self._check_timestamps()
        if not valid:
            return False, start_time

        # Check data size
        auth_size = self.auth_file.stat().st_size / (1024**3)  # GB
        logger.info(f"📊 Auth file size: {auth_size:.1f} GB")

        if auth_size < 0.1:  # Less than 100MB
            logger.warning("⚠️ Auth file seems very small")

        logger.info("✅ Validation complete")
        return True, start_time


class LANLLoader:
    """Load and preprocess LANL cyber security dataset"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.start_date = None  # Will be set after validation

        # Validate and detect start time
        validator = LANLValidator(data_dir)
        passed, detected_start = validator._check_timestamps()

        if not passed:
            raise ValueError("Timestamp validation failed")

        self.start_date = detected_start  # Store it
        logger.info(f"📅 Using start date: {self.start_date}")

        self.auth_file = self.data_dir / "auth.txt"
        self.auth_gz_file = self.data_dir / "auth.txt.gz"
        self.redteam_file = self.data_dir / "redteam.txt"

    def load_sample(self, days: Optional[List[int]] = None, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load authentication and red team data

        Args:
            days: List of days to load (1-58). If None, load all.
            max_rows: Maximum rows to load (for testing)

        Returns:
            Tuple of (auth_df, redteam_df)
        """
        logger.info(f"Loading LANL data from {self.data_dir}")

        # Load authentication data
        auth_df = self._load_auth_data()
        logger.info(f"Loaded {len(auth_df):,} auth events")

        # Load red team data
        redteam_df = self._load_redteam_data()
        if len(redteam_df) > 0:
            logger.info(f"Loaded {len(redteam_df)} red team events")
        else:
            logger.warning("No red team data found")

        # Filter by days if specified
        if days is not None:
            auth_df = self._filter_by_days(auth_df, days)
            redteam_df = self._filter_by_days(redteam_df, days)
            logger.info(f"Filtered to days {min(days)}-{max(days)}: {len(auth_df):,} auth, {len(redteam_df)} red team")

        # Limit rows if specified (for testing)
        if max_rows is not None:
            auth_df = auth_df.head(max_rows).copy()
            logger.info(f"Limited to {max_rows:,} rows for testing")

        return auth_df, redteam_df

    def _load_auth_data(self) -> pd.DataFrame:
        """Load authentication events"""
        # Try gzipped first, then uncompressed
        if self.auth_gz_file.exists():
            logger.info(f"Loading from {self.auth_gz_file}")
            with gzip.open(self.auth_gz_file, 'rt') as f:
                df = self._parse_auth_file(f)
        elif self.auth_file.exists():
            logger.info(f"Loading from {self.auth_file}")
            with open(self.auth_file, 'r') as f:
                df = self._parse_auth_file(f)
        else:
            raise FileNotFoundError(f"Auth file not found: {self.auth_file} or {self.auth_gz_file}")

        return df

    def _parse_auth_file(self, file_handle) -> pd.DataFrame:
        """Parse authentication file into DataFrame"""
        rows = []

        for line_num, line in enumerate(file_handle):
            if line_num % 1000000 == 0 and line_num > 0:
                logger.info(f"Parsed {line_num:,} lines...")

            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            try:
                # Handle LANL format: time (seconds), user_id, src_comp_id, dst_comp_id, auth_type, outcome
                # Convert time from seconds to timestamp using the start_date from validation
                timestamp_seconds = int(parts[0])
                timestamp = self.start_date + pd.Timedelta(seconds=timestamp_seconds)

                row = {
                    'timestamp': timestamp,
                    'user_id': int(parts[1]),
                    'src_computer': parts[2],
                    'dst_computer': parts[3],
                    'auth_type': parts[4],
                    'logon_type': parts[4],  # Use auth_type as logon_type for now
                    'auth_orientation': 'LogOn',  # Default value
                    'outcome': parts[5]
                }
                rows.append(row)

            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
                continue

        df = pd.DataFrame(rows)

        # Add day column
        df['day'] = df['timestamp'].dt.dayofyear

        # Basic validation
        if len(df) == 0:
            raise ValueError("No valid auth events found")

        logger.info(f"Successfully parsed {len(df):,} auth events")
        return df

    def _load_redteam_data(self) -> pd.DataFrame:
        """Load red team attack labels"""
        if not self.redteam_file.exists():
            logger.warning(f"Red team file not found: {self.redteam_file}")
            return pd.DataFrame(columns=['timestamp', 'user_id', 'action', 'day'])

        logger.info(f"Loading red team data from {self.redteam_file}")

        try:
            df = pd.read_csv(self.redteam_file, header=None, names=['timestamp', 'user_id', 'action'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S')
            df['day'] = df['timestamp'].dt.dayofyear

            logger.info(f"Loaded {len(df)} red team events")
            return df

        except Exception as e:
            logger.error(f"Failed to load red team data: {e}")
            return pd.DataFrame(columns=['timestamp', 'user_id', 'action', 'day'])

    def _filter_by_days(self, df: pd.DataFrame, days: List[int]) -> pd.DataFrame:
        """Filter DataFrame to specific days"""
        return df[df['day'].isin(days)].copy()

    def validate_data(self) -> dict:
        """Validate loaded data quality"""
        logger.info("Validating data quality...")

        auth_df, redteam_df = self.load_sample()

        issues = []

        # Check for missing values
        auth_missing = auth_df.isnull().sum()
        if auth_missing.any():
            issues.append(f"Auth data missing values: {auth_missing[auth_missing > 0].to_dict()}")

        # Check for reasonable date range
        if len(auth_df) > 0:
            date_range = auth_df['timestamp'].max() - auth_df['timestamp'].min()
            if date_range.days < 1:
                issues.append(f"Auth data spans only {date_range.days} days")

        # Check for reasonable user count
        if len(auth_df) > 0:
            unique_users = auth_df['user_id'].nunique()
            if unique_users < 10:
                issues.append(f"Only {unique_users} unique users found")

        # Check red team data
        if len(redteam_df) == 0:
            issues.append("No red team data found")

        # Generate report
        report = {
            'auth_events': len(auth_df),
            'redteam_events': len(redteam_df),
            'unique_users': auth_df['user_id'].nunique() if len(auth_df) > 0 else 0,
            'date_range_days': (auth_df['timestamp'].max() - auth_df['timestamp'].min()).days if len(auth_df) > 0 else 0,
            'issues': issues,
            'is_valid': len(issues) == 0
        }

        logger.info(f"Validation report: {report}")
        return report


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

    def validate_timeout(self, events_df: pd.DataFrame):
        """Validate timeout parameter against actual data"""

        gaps = events_df.groupby('user_id')['timestamp'].diff()
        gap_seconds = gaps.dt.total_seconds()

        percentiles = gap_seconds.quantile([0.5, 0.75, 0.9, 0.95, 0.99])

        logger.info("\n📊 Inter-event gaps:")
        logger.info(f"  Median: {percentiles[0.5]/60:.1f} min")
        logger.info(f"  75th percentile: {percentiles[0.75]/60:.1f} min")
        logger.info(f"  95th percentile: {percentiles[0.95]/60:.1f} min")

        # Recommendation
        recommended_timeout = percentiles[0.95] / 60  # 95th percentile in minutes

        if abs(recommended_timeout - self.config.timeout_minutes) > 10:
            logger.warning(f"⚠️  Timeout {self.config.timeout_minutes}min may be suboptimal")
            logger.warning(f"   Recommended: {recommended_timeout:.0f}min (95th percentile)")

    def build_sessions(self,
                       events_df: pd.DataFrame,
                       redteam_df: Optional[pd.DataFrame] = None,
                       train_mode: bool = False) -> List[dict]:  # ADDED train_mode!
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

    def _build_user_sessions(self, user_events: pd.DataFrame, user_id: int) -> List[dict]:
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
        for session_id, session_events in user_events.groupby(session_ids):

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

    def _label_sessions(self, sessions: List[dict], redteam_df: pd.DataFrame) -> List[dict]:
        """Label sessions based on strategy"""

        if self.config.labeling == "strict":
            return self._label_strict(sessions, redteam_df)
        elif self.config.labeling == "window":
            return self._label_window(sessions, redteam_df)
        elif self.config.labeling == "user_day":
            return self._label_user_day(sessions, redteam_df)
        else:
            raise ValueError(f"Unknown labeling: {self.config.labeling}")

    def _label_window(self, sessions: List[dict], redteam_df: pd.DataFrame) -> List[dict]:
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

    def _label_strict(self, sessions: List[dict], redteam_df: pd.DataFrame) -> List[dict]:
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

    def _label_user_day(self, sessions: List[dict], redteam_df: pd.DataFrame) -> List[dict]:
        """Label all sessions by user on attack day"""

        attack_user_days = set(zip(redteam_df['user_id'], redteam_df['timestamp'].dt.date))

        for session in sessions:
            user_id = session['user_id']
            session_date = session['start_time'].date()

            if (user_id, session_date) in attack_user_days:
                session['is_malicious'] = True

        return sessions
