"""
Storage Manager: Centralized data and model persistence layer.

Responsibilities:
- Handle model serialization and storage
- Manage prediction logs
- Persist decision records
- Handle drift detection logs
- Cleanup and archival
"""

import json
import logging
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

try:
    from configs.settings import (
        MODELS_PATH, LOGS_PATH, get_log_file,
        PREDICTION_LOG_FILE, DRIFT_LOG_FILE
    )
except ImportError:
    MODELS_PATH = Path("./storage/models")
    LOGS_PATH = Path("./storage/logs")
    def get_log_file(component):
        return str(LOGS_PATH / f"{component}.log")
    PREDICTION_LOG_FILE = str(LOGS_PATH / "predictions.log")
    DRIFT_LOG_FILE = str(LOGS_PATH / "drift_detection.log")


logger = logging.getLogger(__name__)


class StorageManager:
    """Manages all data persistence for INTERPRETX."""
    
    def __init__(self):
        """Initialize storage manager and ensure directories exist."""
        self.models_path = MODELS_PATH
        self.logs_path = LOGS_PATH
        
        # Create directories if they don't exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.model_registry_path = self.models_path / "registry"
        self.predictions_log_path = self.logs_path / "predictions"
        self.drift_log_path = self.logs_path / "drift_detection"
        self.decision_log_path = self.logs_path / "decisions"
        
        for path in [self.model_registry_path, self.predictions_log_path, 
                     self.drift_log_path, self.decision_log_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage manager initialized. Models: {self.models_path}, Logs: {self.logs_path}")
    
    # ==================== MODEL STORAGE ====================
    
    def save_model(self, model: Any, model_id: str, version: str) -> Path:
        """
        Save a model to disk.
        
        Args:
            model: The model object to save
            model_id: Model identifier
            version: Model version
            
        Returns:
            Path where model was saved
        """
        import joblib
        
        model_file = self.model_registry_path / f"{model_id}_v{version}.pkl"
        
        try:
            joblib.dump(model, model_file)
            logger.info(f"Model {model_id} v{version} saved to {model_file}")
            return model_file
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            raise
    
    def load_model(self, model_id: str, version: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            The loaded model object
        """
        import joblib
        
        model_file = self.model_registry_path / f"{model_id}_v{version}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            model = joblib.load(model_file)
            logger.info(f"Model {model_id} v{version} loaded from {model_file}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    # ==================== PREDICTION LOGGING ====================
    
    def log_prediction(self, prediction_result: Any) -> Path:
        """
        Log a prediction result.
        
        Args:
            prediction_result: PredictionResult object (must have to_dict method)
            
        Returns:
            Path where prediction was logged
        """
        log_file = self.logs_path / "predictions.jsonl"
        
        try:
            # Convert to dict if it has to_dict method
            if hasattr(prediction_result, 'to_dict'):
                data = prediction_result.to_dict()
            else:
                data = prediction_result if isinstance(prediction_result, dict) else asdict(prediction_result)
            
            # Append to JSONL file
            with open(log_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')
            
            return log_file
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            raise
    
    def get_prediction_logs(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve prediction logs.
        
        Args:
            model_id: Optional filter by model
            limit: Maximum number of logs to retrieve
            
        Returns:
            List of prediction logs
        """
        log_file = self.logs_path / "predictions.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        log = json.loads(line)
                        if model_id is None or log.get('model_id') == model_id:
                            logs.append(log)
        except Exception as e:
            logger.error(f"Error reading prediction logs: {e}")
        
        return logs[-limit:]
    
    # ==================== DECISION LOGGING ====================
    
    def log_decision(self, decision_record: Dict[str, Any]) -> Path:
        """
        Log a governance decision.
        
        Args:
            decision_record: Dictionary with decision details
            
        Returns:
            Path where decision was logged
        """
        log_file = self.logs_path / "decisions.jsonl"
        
        # Ensure timestamp
        if 'timestamp' not in decision_record:
            decision_record['timestamp'] = datetime.now().isoformat()
        
        try:
            with open(log_file, 'a') as f:
                json.dump(decision_record, f)
                f.write('\n')
            
            logger.info(f"Decision logged: {decision_record.get('decision_id', 'unknown')}")
            return log_file
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            raise
    
    def get_decision_logs(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve decision logs.
        
        Args:
            limit: Maximum number of logs
            
        Returns:
            List of decision logs
        """
        log_file = self.logs_path / "decisions.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading decision logs: {e}")
        
        return logs[-limit:]
    
    # ==================== DRIFT DETECTION LOGGING ====================
    
    def log_drift_detection(self, drift_record: Dict[str, Any]) -> Path:
        """
        Log drift detection event.
        
        Args:
            drift_record: Dictionary with drift details
            
        Returns:
            Path where drift event was logged
        """
        log_file = self.logs_path / "drift_detection.jsonl"
        
        # Ensure timestamp
        if 'timestamp' not in drift_record:
            drift_record['timestamp'] = datetime.now().isoformat()
        
        try:
            with open(log_file, 'a') as f:
                json.dump(drift_record, f)
                f.write('\n')
            
            logger.warning(f"Drift detected: {drift_record.get('drift_type', 'unknown')}")
            return log_file
        except Exception as e:
            logger.error(f"Error logging drift detection: {e}")
            raise
    
    def get_drift_logs(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve drift detection logs.
        
        Args:
            limit: Maximum number of logs
            
        Returns:
            List of drift logs
        """
        log_file = self.logs_path / "drift_detection.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading drift logs: {e}")
        
        return logs[-limit:]
    
    # ==================== METADATA & REGISTRY ====================
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str) -> Path:
        """
        Save metadata JSON file.
        
        Args:
            metadata: Dictionary with metadata
            filename: Filename to save as
            
        Returns:
            Path where metadata was saved
        """
        file_path = self.logs_path / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def load_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Load metadata JSON file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Metadata dictionary
        """
        file_path = self.logs_path / filename
        
        if not file_path.exists():
            logger.warning(f"Metadata file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Metadata loaded from {file_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    # ==================== STORAGE STATISTICS ====================
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        def get_dir_size(path: Path) -> int:
            """Calculate total size of directory."""
            total = 0
            if path.exists():
                for f in path.rglob('*'):
                    if f.is_file():
                        total += f.stat().st_size
            return total
        
        def count_files(path: Path, pattern: str = '*') -> int:
            """Count files in directory."""
            if path.exists():
                return len(list(path.glob(pattern)))
            return 0
        
        stats = {
            'models_dir': {
                'size_bytes': get_dir_size(self.model_registry_path),
                'file_count': count_files(self.model_registry_path, '*.pkl'),
            },
            'logs_dir': {
                'size_bytes': get_dir_size(self.logs_path),
                'prediction_logs': count_files(Path(LOGS_PATH / "predictions.jsonl")),
                'decision_logs': count_files(Path(LOGS_PATH / "decisions.jsonl")),
                'drift_logs': count_files(Path(LOGS_PATH / "drift_detection.jsonl")),
            },
            'total_size_bytes': get_dir_size(self.models_path) + get_dir_size(self.logs_path),
            'timestamp': datetime.now().isoformat(),
        }
        
        return stats
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        Clean up logs older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of files deleted
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        try:
            for log_file in self.logs_path.rglob('*.jsonl'):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old log: {log_file}")
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
        
        return deleted_count
    
    def verify_storage(self) -> bool:
        """
        Verify storage setup is correct.
        
        Returns:
            True if all checks pass
        """
        checks = [
            (self.models_path.exists() and self.models_path.is_dir(), "Models directory exists"),
            (self.logs_path.exists() and self.logs_path.is_dir(), "Logs directory exists"),
            (self.model_registry_path.exists(), "Model registry subdirectory exists"),
            (self.decision_log_path.exists(), "Decision log subdirectory exists"),
            (self.drift_log_path.exists(), "Drift log subdirectory exists"),
        ]
        
        all_valid = True
        for check, message in checks:
            if check:
                logger.info(f"✓ {message}")
            else:
                logger.error(f"✗ {message}")
                all_valid = False
        
        return all_valid
