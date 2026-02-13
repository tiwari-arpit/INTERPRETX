"""
Prediction Engine: Core prediction and inference system for INTERPRETX.

Responsibilities:
- Execute predictions on input data
- Provide confidence/uncertainty quantification
- Validate input features
- Track prediction history and performance
- Provide decision signals for governance layer
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import storage manager for logging predictions
try:
    from core.storage_manager import StorageManager
except ImportError:
    StorageManager = None


logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level classification."""
    HIGH = "high"  # >= 0.85
    MEDIUM = "medium"  # 0.65 - 0.85
    LOW = "low"  # 0.50 - 0.65
    CRITICAL = "critical"  # < 0.50


class PredictionType(Enum):
    """Type of prediction."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS = "multi_class"
    REGRESSION = "regression"
    PROBABILITY = "probability"


@dataclass
class PredictionResult:
    """Result of a prediction."""
    prediction_id: str
    model_id: str
    input_features: Dict[str, Any]
    prediction: Union[float, int, np.ndarray]
    confidence: float
    confidence_level: ConfidenceLevel
    
    # Uncertainty metrics
    prediction_std: Optional[float] = None  # Standard deviation if available
    prediction_interval: Optional[Tuple[float, float]] = None  # Confidence interval
    
    # Additional signals
    is_out_of_distribution: bool = False
    feature_drift_detected: bool = False
    prediction_drift_detected: bool = False
    
    # Metadata
    prediction_type: str = "unknown"
    model_version: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_id': self.prediction_id,
            'model_id': self.model_id,
            'input_features': self.input_features,
            'prediction': self._serialize_value(self.prediction),
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'prediction_std': self.prediction_std,
            'prediction_interval': self.prediction_interval,
            'is_out_of_distribution': self.is_out_of_distribution,
            'feature_drift_detected': self.feature_drift_detected,
            'prediction_drift_detected': self.prediction_drift_detected,
            'prediction_type': self.prediction_type,
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'execution_time_ms': self.execution_time_ms,
        }
    
    @staticmethod
    def _serialize_value(value):
        """Serialize value for JSON compatibility."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        return value


class PredictionEngine:
    """Core inference and prediction system."""
    
    def __init__(
        self,
        model_registry: Any,
        ensemble_manager: Optional[Any] = None,
        enable_uncertainty: bool = True,
        storage_manager: Optional[Any] = None,
    ):
        """
        Initialize prediction engine.
        
        Args:
            model_registry: Reference to ModelRegistry
            ensemble_manager: Optional reference to EnsembleManager
            enable_uncertainty: Whether to compute uncertainty estimates
            storage_manager: Optional StorageManager for logging predictions
        """
        self.model_registry = model_registry
        self.ensemble_manager = ensemble_manager
        self.enable_uncertainty = enable_uncertainty
        
        # Initialize storage manager if provided
        if storage_manager is None and StorageManager:
            try:
                self.storage_manager = StorageManager()
            except Exception as e:
                logger.warning(f"Could not initialize StorageManager: {e}")
                self.storage_manager = None
        else:
            self.storage_manager = storage_manager
        
        # Prediction history tracking
        self.prediction_history: List[PredictionResult] = []
        self.max_history_size = 10000  # Keep last N predictions
        
        # Drift detection baseline (learned from initial predictions)
        self.feature_baseline: Optional[Dict[str, Dict[str, float]]] = None
        self.prediction_baseline: Optional[Dict[str, Dict[str, float]]] = None
    
    def predict(
        self,
        model_id: str,
        features: Union[Dict, pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        return_uncertainty: bool = True,
    ) -> PredictionResult:
        """
        Make a prediction with a specific model.
        
        Args:
            model_id: ID of the model to use
            features: Input features (dict, DataFrame, or array)
            feature_names: Names of features if using array format
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            PredictionResult with prediction and metadata
        """
        import time
        start_time = time.time()
        
        # Convert features to standard format
        features_dict, features_array, feature_names = self._prepare_features(
            features, feature_names
        )
        
        # Validate features
        try:
            self._validate_features(model_id, features_dict, feature_names)
        except ValueError as e:
            logger.warning(f"Feature validation warning: {e}")
        
        # Retrieve model
        try:
            model, metadata = self.model_registry.get_model(model_id)
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {e}")
            raise
        
        # Make prediction
        try:
            pred = model.predict(features_array)
            
            # Get confidence score
            confidence = self._get_confidence(model, features_array, pred)
            
            # Compute uncertainty if available
            uncertainty_info = {}
            if return_uncertainty and self.enable_uncertainty:
                uncertainty_info = self._estimate_uncertainty(model, features_array, pred)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
        
        # Detect out-of-distribution
        ood = self._detect_ood(features_dict)
        
        # Detect drift
        feature_drift = self._detect_feature_drift(features_dict)
        pred_drift = self._detect_prediction_drift(model_id, pred)
        
        # Determine confidence level
        conf_level = self._classify_confidence(confidence)
        
        # Create result
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = PredictionResult(
            prediction_id=self._generate_prediction_id(),
            model_id=model_id,
            input_features=features_dict,
            prediction=pred,
            confidence=float(confidence),
            confidence_level=conf_level,
            prediction_std=uncertainty_info.get('std'),
            prediction_interval=uncertainty_info.get('interval'),
            is_out_of_distribution=ood,
            feature_drift_detected=feature_drift,
            prediction_drift_detected=pred_drift,
            prediction_type=metadata.output_type,
            model_version=metadata.version,
            execution_time_ms=float(execution_time),
        )
        
        # Store in history
        self._add_to_history(result)
        
        logger.info(f"Prediction made with confidence: {confidence:.3f}, OOD: {ood}")
        
        return result
    
    def ensemble_predict(
        self,
        ensemble_id: str,
        features: Union[Dict, pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Make prediction using ensemble.
        
        Args:
            ensemble_id: ID of ensemble to use
            features: Input features
            feature_names: Names of features
            
        Returns:
            PredictionResult from ensemble
        """
        if not self.ensemble_manager:
            raise RuntimeError("Ensemble manager not configured")
        
        # Convert features
        features_dict, features_array, feature_names = self._prepare_features(
            features, feature_names
        )
        
        # Get ensemble prediction
        ensemble_result = self.ensemble_manager.predict(ensemble_id, features_array)
        
        # Convert to PredictionResult
        result = PredictionResult(
            prediction_id=self._generate_prediction_id(),
            model_id=f"ensemble_{ensemble_id}",
            input_features=features_dict,
            prediction=ensemble_result.prediction,
            confidence=ensemble_result.confidence,
            confidence_level=self._classify_confidence(ensemble_result.confidence),
            is_out_of_distribution=self._detect_ood(features_dict),
            feature_drift_detected=self._detect_feature_drift(features_dict),
            prediction_type="ensemble",
            execution_time_ms=0.0,
        )
        
        self._add_to_history(result)
        
        return result
    
    def batch_predict(
        self,
        model_id: str,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        Make predictions for a batch of samples.
        
        Args:
            model_id: Model to use
            features: Batch of features
            feature_names: Feature names
            
        Returns:
            List of PredictionResults
        """
        results = []
        
        if isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
            features_array = features.values
        else:
            features_array = np.asarray(features)
        
        for i, sample in enumerate(features_array):
            sample_dict = {}
            if feature_names:
                sample_dict = dict(zip(feature_names, sample))
            
            result = self.predict(model_id, sample_dict, feature_names)
            results.append(result)
        
        return results
    
    def _prepare_features(
        self,
        features: Union[Dict, pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Dict, np.ndarray, List[str]]:
        """Convert various feature formats to standard formats."""
        if isinstance(features, dict):
            features_dict = features
            features_array = np.array([list(features.values())])
            feature_names = list(features.keys())
        
        elif isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
            features_dict = dict(zip(feature_names, features.iloc[0]))
            features_array = features.values
        
        else:
            features_array = np.asarray(features)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            if feature_names:
                features_dict = dict(zip(feature_names, features_array[0]))
            else:
                features_dict = {f"feature_{i}": v for i, v in enumerate(features_array[0])}
        
        return features_dict, features_array, feature_names or []
    
    def _validate_features(
        self,
        model_id: str,
        features_dict: Dict[str, Any],
        feature_names: List[str],
    ) -> None:
        """Validate features against model expectations."""
        try:
            metadata = self.model_registry.get_metadata(model_id)
            
            expected_features = set(metadata.input_features)
            provided_features = set(features_dict.keys())
            
            missing = expected_features - provided_features
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
        except Exception as e:
            logger.warning(f"Feature validation failed: {e}")
    
    def _get_confidence(self, model: Any, features: np.ndarray, prediction: Any) -> float:
        """Extract confidence score from model."""
        # Try to get probability predictions
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)
                return float(np.max(proba))
            except:
                pass
        
        # Try to get decision function
        if hasattr(model, 'decision_function'):
            try:
                decision = model.decision_function(features)
                return float(1.0 / (1.0 + np.exp(-decision)))  # Sigmoid
            except:
                pass
        
        # Default confidence based on model type
        return 0.5
    
    def _estimate_uncertainty(
        self,
        model: Any,
        features: np.ndarray,
        prediction: Any
    ) -> Dict[str, Any]:
        """Estimate prediction uncertainty."""
        uncertainty = {}
        
        # Try to get prediction std if available (e.g., from ensemble or Bayesian model)
        if hasattr(model, 'predict_std'):
            try:
                std = model.predict_std(features)[0]
                uncertainty['std'] = float(std)
                
                # Compute confidence interval
                pred_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
                uncertainty['interval'] = (
                    pred_value - 1.96 * std,
                    pred_value + 1.96 * std
                )
            except:
                pass
        
        return uncertainty
    
    def _detect_ood(self, features: Dict[str, Any]) -> bool:
        """Detect if features are out-of-distribution."""
        # Placeholder for OOD detection
        # In full implementation, compare against training data distribution
        return False
    
    def _detect_feature_drift(self, features: Dict[str, Any]) -> bool:
        """Detect if features show drift from baseline."""
        if not self.feature_baseline:
            self._update_baseline(features, is_feature_baseline=True)
            return False
        
        # Placeholder for drift detection
        return False
    
    def _detect_prediction_drift(self, model_id: str, prediction: Any) -> bool:
        """Detect if predictions show drift from baseline."""
        if not self.prediction_baseline:
            if model_id not in self.prediction_baseline:
                self.prediction_baseline = {model_id: {'mean': float(prediction), 'count': 1}}
            return False
        
        # Placeholder for drift detection
        return False
    
    def _classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Classify confidence score into levels."""
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.65:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.CRITICAL
    
    def _update_baseline(
        self,
        features: Dict[str, Any],
        is_feature_baseline: bool = True
    ) -> None:
        """Update baseline for drift detection."""
        if is_feature_baseline:
            self.feature_baseline = {
                name: {'mean': float(value), 'min': float(value), 'max': float(value)}
                for name, value in features.items()
            }
        else:
            if not self.prediction_baseline:
                self.prediction_baseline = {}
    
    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _add_to_history(self, result: PredictionResult) -> None:
        """Add prediction to history, removing old entries if needed."""
        self.prediction_history.append(result)
        
        # Log to storage if available
        if self.storage_manager:
            try:
                self.storage_manager.log_prediction(result)
            except Exception as e:
                logger.warning(f"Could not log prediction to storage: {e}")
        
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history = self.prediction_history[-self.max_history_size:]
    
    def get_prediction_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PredictionResult]:
        """
        Retrieve prediction history.
        
        Args:
            model_id: Filter by model (None = all)
            limit: Maximum results
            
        Returns:
            List of PredictionResults
        """
        history = self.prediction_history
        
        if model_id:
            history = [p for p in history if p.model_id == model_id]
        
        return history[-limit:]
    
    def get_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics on predictions.
        
        Args:
            model_id: Filter by model
            
        Returns:
            Statistics dictionary
        """
        predictions = self.get_prediction_history(model_id)
        
        if not predictions:
            return {}
        
        confidences = [p.confidence for p in predictions]
        
        stats = {
            'total_predictions': len(predictions),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'std_confidence': float(np.std(confidences)),
            'ood_count': sum(1 for p in predictions if p.is_out_of_distribution),
            'drift_detected_count': sum(1 for p in predictions if p.feature_drift_detected),
        }
        
        return stats
