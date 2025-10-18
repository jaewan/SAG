# ====================================================================================
# FILE: src/utils/experiment_tracker.py
# ====================================================================================

"""
Experiment Tracking System for Phase 1

Features:
- Structured logging (JSONL format)
- Reproducibility tracking (config, code version, environment)
- Metric logging with timestamps
- Hypothesis testing results
- Run comparison
- Integration with popular ML tools (optional)

Design Philosophy:
- Simple, file-based storage (no database required)
- Easy to parse and analyze
- Git-friendly (text-based formats)
- Production-ready with minimal dependencies
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import socket
import sys
import os
import pickle

logger = logging.getLogger(__name__)


class ExperimentRun:
    """
    Represents a single experiment run with all metadata

    Stores:
    - Configuration
    - Environment (Python version, packages)
    - Code version (git commit)
    - System info (hostname, RAM)
    - Metrics over time
    - Final results
    """

    def __init__(self, experiment_name: str, run_id: str = None):
        self.experiment_name = experiment_name
        self.run_id = run_id or self._generate_run_id()
        self.start_time = datetime.now()
        self.end_time = None

        # Metadata
        self.config = {}
        self.environment = self._capture_environment()
        self.system_info = self._capture_system_info()
        self.git_info = self._capture_git_info()

        # Results
        self.stages = {}
        self.metrics = []
        self.artifacts = {}
        self.status = "running"
        self.error = None

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"{timestamp}_{random_suffix}"

    def _capture_environment(self) -> Dict:
        """Capture Python environment"""
        try:
            import pkg_resources
            packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        except:
            packages = {}

        return {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'packages': packages
        }

    def _capture_system_info(self) -> Dict:
        """Capture system information"""
        import psutil

        return {
            'hostname': socket.gethostname(),
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

    def _capture_git_info(self) -> Dict:
        """Capture git repository information"""
        try:
            import subprocess

            def run_git_command(cmd):
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent
                )
                return result.stdout.strip() if result.returncode == 0 else None

            return {
                'commit_hash': run_git_command('git rev-parse HEAD'),
                'branch': run_git_command('git rev-parse --abbrev-ref HEAD'),
                'commit_message': run_git_command('git log -1 --pretty=%B'),
                'is_dirty': run_git_command('git diff --quiet') is None,
                'remote_url': run_git_command('git config --get remote.origin.url')
            }
        except:
            return {'error': 'Git information not available'}

    def set_config(self, config: Dict):
        """Set experiment configuration"""
        self.config = config

    def log_stage(self, stage_name: str, metrics: Dict):
        """Log completion of a stage"""
        self.stages[stage_name] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

    def log_metric(self, name: str, value: Any, step: int = None):
        """Log a metric value"""
        self.metrics.append({
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'step': step
        })

    def add_artifact(self, name: str, path: str, artifact_type: str = "file"):
        """Register an artifact"""
        self.artifacts[name] = {
            'path': str(path),
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        }

    def mark_complete(self, status: str = "completed"):
        """Mark run as complete"""
        self.end_time = datetime.now()
        self.status = status

    def mark_failed(self, error: str):
        """Mark run as failed"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error = error

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'status': self.status,
            'error': self.error,
            'config': self.config,
            'environment': self.environment,
            'system_info': self.system_info,
            'git_info': self.git_info,
            'stages': self.stages,
            'metrics': self.metrics,
            'artifacts': self.artifacts
        }


class ExperimentTracker:
    """
    Main experiment tracking interface

    Usage:
        tracker = ExperimentTracker("phase1_experiment", output_dir="experiments/")
        tracker.log_experiment_start(config)
        tracker.log_stage("data_loading", {"events": 1000000})
        tracker.log_metric("memory_mb", 1500)
        tracker.log_decision({"verdict": "PROCEED"})
        tracker.log_experiment_end()
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create run
        self.run = ExperimentRun(experiment_name)

        # Create run directory
        self.run_dir = self.output_dir / self.run.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.metadata_file = self.run_dir / "metadata.json"
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.stages_file = self.run_dir / "stages.jsonl"

        logger.info(f"📊 Experiment tracker initialized: {self.run.run_id}")
        logger.info(f"   Output directory: {self.run_dir}")

    def log_experiment_start(self, config: Dict):
        """Log experiment start with configuration"""
        logger.info(f"📊 Experiment started: {self.experiment_name}")

        self.run.set_config(config)

        # Save initial metadata
        self._save_metadata()

        # Log to JSONL
        self._append_to_jsonl(self.stages_file, {
            'event': 'experiment_start',
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run.run_id,
            'config_hash': self._hash_dict(config)
        })

    def log_stage(self, stage_name: str, metrics: Dict):
        """Log stage completion"""
        logger.info(f"📊 Stage complete: {stage_name}")

        self.run.log_stage(stage_name, metrics)

        # Save to stages log
        self._append_to_jsonl(self.stages_file, {
            'event': 'stage_complete',
            'timestamp': datetime.now().isoformat(),
            'stage': stage_name,
            'metrics': metrics
        })

        # Update metadata
        self._save_metadata()

    def log_metric(self, name: str, value: Any, step: int = None):
        """Log a metric"""
        self.run.log_metric(name, value, step)

        # Save to metrics log
        self._append_to_jsonl(self.metrics_file, {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'step': step
        })

    def log_decision(self, decision: Dict):
        """Log final proceed/stop decision"""
        logger.info(f"📊 Decision logged: {decision.get('verdict')}")

        self.run.log_stage('decision', decision)

        # Save decision as artifact
        decision_file = self.run_dir / "decision.json"
        with open(decision_file, 'w') as f:
            json.dump(decision, f, indent=2)

        self.run.add_artifact('decision', decision_file, 'json')

    def log_experiment_end(self, status: str = "completed"):
        """Log experiment completion"""
        logger.info(f"📊 Experiment ended: {status}")

        self.run.mark_complete(status)
        self._save_metadata()

        # Log to JSONL
        self._append_to_jsonl(self.stages_file, {
            'event': 'experiment_end',
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'duration_seconds': (self.run.end_time - self.run.start_time).total_seconds()
        })

        # Generate summary
        self._generate_summary()

    def log_error(self, error: str):
        """Log error and mark as failed"""
        logger.error(f"📊 Experiment failed: {error}")

        self.run.mark_failed(error)
        self._save_metadata()

    def add_artifact(self, name: str, path: Path, artifact_type: str = "file"):
        """Add artifact to tracking"""
        self.run.add_artifact(name, path, artifact_type)
        self._save_metadata()

    def _save_metadata(self):
        """Save run metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.run.to_dict(), f, indent=2)

    def _append_to_jsonl(self, file_path: Path, data: Dict):
        """Append to JSONL file"""
        with open(file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _hash_dict(self, d: Dict) -> str:
        """Create hash of dictionary"""
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def _generate_summary(self):
        """Generate human-readable summary"""
        summary_file = self.run_dir / "summary.txt"

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Run ID: {self.run.run_id}\n")
            f.write(f"Status: {self.run.status}\n")
            f.write(f"Start: {self.run.start_time}\n")
            f.write(f"End: {self.run.end_time}\n")
            if self.run.end_time:
                duration = (self.run.end_time - self.start_time).total_seconds()
                f.write(f"Duration: {duration/3600:.2f} hours\n")
            f.write("\n")

            # System info
            f.write("System Information:\n")
            f.write(f"  Hostname: {self.run.system_info['hostname']}\n")
            f.write(f"  CPUs: {self.run.system_info['cpu_count']}\n")
            f.write(f"  Total RAM: {self.run.system_info['total_memory_gb']:.1f}GB\n")
            f.write("\n")

            # Git info
            f.write("Git Information:\n")
            if 'commit_hash' in self.run.git_info:
                f.write(f"  Commit: {self.run.git_info['commit_hash']}\n")
                f.write(f"  Branch: {self.run.git_info['branch']}\n")
                f.write(f"  Dirty: {self.run.git_info['is_dirty']}\n")
            f.write("\n")

            # Stages
            f.write("Stages Completed:\n")
            for stage_name, stage_data in self.run.stages.items():
                f.write(f"  {stage_name}:\n")
                for key, value in stage_data['metrics'].items():
                    f.write(f"    {key}: {value}\n")
            f.write("\n")

            # Artifacts
            f.write("Artifacts:\n")
            for name, artifact in self.run.artifacts.items():
                f.write(f"  {name}: {artifact['path']}\n")

            f.write("\n" + "="*80 + "\n")

        logger.info(f"📄 Summary saved to {summary_file}")


class ExperimentComparator:
    """
    Compare multiple experiment runs

    Usage:
        comparator = ExperimentComparator()
        comparator.load_runs(["experiments/run1", "experiments/run2"])
        comparator.compare_metrics()
        comparator.generate_report()
    """

    def __init__(self):
        self.runs = []

    def load_run(self, run_dir: Path):
        """Load a single run"""
        metadata_file = Path(run_dir) / "metadata.json"

        if not metadata_file.exists():
            logger.warning(f"Metadata not found: {metadata_file}")
            return

        with open(metadata_file, 'r') as f:
            run_data = json.load(f)

        self.runs.append(run_data)
        logger.info(f"Loaded run: {run_data['run_id']}")

    def load_runs(self, run_dirs: List[Path]):
        """Load multiple runs"""
        for run_dir in run_dirs:
            self.load_run(run_dir)

    def load_experiment(self, experiment_dir: Path):
        """Load all runs from an experiment directory"""
        experiment_path = Path(experiment_dir)

        for run_dir in experiment_path.iterdir():
            if run_dir.is_dir():
                self.load_run(run_dir)

    def compare_configs(self) -> Dict:
        """Compare configurations across runs"""
        if len(self.runs) < 2:
            logger.warning("Need at least 2 runs to compare")
            return {}

        # Find config differences
        configs = [run['config'] for run in self.runs]
        differences = self._find_dict_differences(configs)

        logger.info(f"Configuration differences across {len(self.runs)} runs:")
        for key, values in differences.items():
            logger.info(f"  {key}: {values}")

        return differences

    def compare_metrics(self) -> Dict:
        """Compare final metrics across runs"""
        comparison = {}

        # Extract metrics from each run
        for run in self.runs:
            run_id = run['run_id']

            # Extract key metrics from stages
            for stage_name, stage_data in run.get('stages', {}).items():
                for metric_name, metric_value in stage_data.get('metrics', {}).items():
                    full_metric_name = f"{stage_name}.{metric_name}"

                    if full_metric_name not in comparison:
                        comparison[full_metric_name] = {}

                    comparison[full_metric_name][run_id] = metric_value

        return comparison

    def find_best_run(self, metric: str, maximize: bool = True) -> Dict:
        """Find best run based on a metric"""
        runs_with_metric = []

        for run in self.runs:
            # Try to extract metric from stages
            metric_value = None
            for stage_name, stage_data in run.get('stages', {}).items():
                if metric in stage_data.get('metrics', {}):
                    metric_value = stage_data['metrics'][metric]
                    break

            if metric_value is not None:
                runs_with_metric.append((run['run_id'], metric_value, run))

        if not runs_with_metric:
            logger.warning(f"Metric '{metric}' not found in any run")
            return {}

        # Find best
        best = max(runs_with_metric, key=lambda x: x[1]) if maximize else min(runs_with_metric, key=lambda x: x[1])

        logger.info(f"Best run for {metric} ({'max' if maximize else 'min'}): {best[0]} = {best[1]}")

        return best[2]

    def generate_comparison_report(self, output_file: Path):
        """Generate comparison report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Number of runs: {len(self.runs)}\n\n")

            # Run summary
            f.write("Runs:\n")
            for run in self.runs:
                f.write(f"  {run['run_id']}: {run['status']}")
                if run.get('duration_seconds'):
                    f.write(f" ({run['duration_seconds']/3600:.2f}h)")
                f.write("\n")
            f.write("\n")

            # Config differences
            f.write("Configuration Differences:\n")
            config_diff = self.compare_configs()
            if config_diff:
                for key, values in config_diff.items():
                    f.write(f"  {key}:\n")
                    for run_id, value in values.items():
                        f.write(f"    {run_id}: {value}\n")
            else:
                f.write("  (All configurations identical)\n")
            f.write("\n")

            # Metric comparison
            f.write("Metric Comparison:\n")
            metrics = self.compare_metrics()
            for metric_name, values in metrics.items():
                f.write(f"  {metric_name}:\n")
                for run_id, value in values.items():
                    f.write(f"    {run_id}: {value}\n")

            f.write("\n" + "="*80 + "\n")

        logger.info(f"📄 Comparison report saved to {output_file}")

    def _find_dict_differences(self, dicts: List[Dict]) -> Dict:
        """Find differences across dictionaries"""
        differences = {}

        # Get all keys
        all_keys = set()
        for d in dicts:
            all_keys.update(self._flatten_dict(d).keys())

        # Check each key
        for key in all_keys:
            values = {}
            for i, d in enumerate(dicts):
                flat_d = self._flatten_dict(d)
                if key in flat_d:
                    values[f"run_{i}"] = flat_d[key]

            # If values differ, record it
            unique_values = set(str(v) for v in values.values())
            if len(unique_values) > 1:
                differences[key] = values

        return differences

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ReproducibilityChecker:
    """
    Check if experiment is reproducible

    Validates:
    - Configuration consistency
    - Random seed usage
    - Code version tracking
    - Environment consistency
    """

    def __init__(self):
        self.checks = []

    def check_run(self, run_dir: Path) -> Dict:
        """Run reproducibility checks on a single run"""
        metadata_file = Path(run_dir) / "metadata.json"

        if not metadata_file.exists():
            return {'error': 'Metadata not found'}

        with open(metadata_file, 'r') as f:
            run_data = json.load(f)

        checks = {
            'has_config': 'config' in run_data and len(run_data['config']) > 0,
            'has_git_commit': 'git_info' in run_data and 'commit_hash' in run_data['git_info'],
            'git_is_clean': run_data.get('git_info', {}).get('is_dirty') == False,
            'has_random_seed': self._check_random_seed(run_data['config']),
            'has_environment': 'environment' in run_data,
            'has_system_info': 'system_info' in run_data
        }

        # Overall score
        score = sum(checks.values()) / len(checks) * 100
        checks['reproducibility_score'] = score

        logger.info(f"Reproducibility score: {score:.1f}%")
        for check_name, passed in checks.items():
            if check_name != 'reproducibility_score':
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {check_name}")

        return checks

    def _check_random_seed(self, config: Dict) -> bool:
        """Check if random seed is set"""
        # Look for seed in config
        flat_config = self._flatten_dict(config)
        seed_keys = [k for k in flat_config.keys() if 'seed' in k.lower()]
        return len(seed_keys) > 0

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ====================================================================================
# Command-line tools for experiment management
# ====================================================================================

def cli_list_experiments(base_dir: Path):
    """List all experiments"""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return

    print("="*80)
    print("EXPERIMENTS")
    print("="*80)

    for exp_dir in sorted(base_path.iterdir()):
        if exp_dir.is_dir():
            # Count runs
            runs = [d for d in exp_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
            print(f"\n{exp_dir.name}:")
            print(f"  Runs: {len(runs)}")

            if runs:
                # Show latest run
                latest = max(runs, key=lambda d: d.stat().st_mtime)
                metadata_file = latest / "metadata.json"
                with open(metadata_file, 'r') as f:
                    run_data = json.load(f)

                print(f"  Latest: {run_data['run_id']}")
                print(f"    Status: {run_data['status']}")
                print(f"    Date: {run_data['start_time']}")


def cli_show_run(run_dir: Path):
    """Show details of a run"""
    metadata_file = Path(run_dir) / "metadata.json"

    if not metadata_file.exists():
        print(f"Run not found: {run_dir}")
        return

    with open(metadata_file, 'r') as f:
        run_data = json.load(f)

    print("="*80)
    print(f"RUN: {run_data['run_id']}")
    print("="*80)

    print(f"\nStatus: {run_data['status']}")
    print(f"Start: {run_data['start_time']}")
    print(f"End: {run_data.get('end_time', 'N/A')}")

    if run_data.get('duration_seconds'):
        print(f"Duration: {run_data['duration_seconds']/3600:.2f} hours")

    print(f"\nStages:")
    for stage_name in run_data.get('stages', {}).keys():
        print(f"  - {stage_name}")

    print(f"\nArtifacts:")
    for name, artifact in run_data.get('artifacts', {}).items():
        print(f"  - {name}: {artifact['path']}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Experiment tracking CLI')
    subparsers = parser.add_subparsers(dest='command')

    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument('--dir', type=str, default='experiments/', help='Base directory')

    # Show run
    show_parser = subparsers.add_parser('show', help='Show run details')
    show_parser.add_argument('run_dir', type=str, help='Run directory')

    # Compare runs
    compare_parser = subparsers.add_parser('compare', help='Compare runs')
    compare_parser.add_argument('run_dirs', type=str, nargs='+', help='Run directories')
    compare_parser.add_argument('--output', type=str, default='comparison.txt', help='Output file')

    # Check reproducibility
    check_parser = subparsers.add_parser('check', help='Check reproducibility')
    check_parser.add_argument('run_dir', type=str, help='Run directory')

    args = parser.parse_args()

    if args.command == 'list':
        cli_list_experiments(args.dir)
    elif args.command == 'show':
        cli_show_run(args.run_dir)
    elif args.command == 'compare':
        comparator = ExperimentComparator()
        comparator.load_runs([Path(d) for d in args.run_dirs])
        comparator.generate_comparison_report(Path(args.output))
    elif args.command == 'check':
        checker = ReproducibilityChecker()
        checker.check_run(Path(args.run_dir))
