"""
Ensemble Manager: Orchestrate predictions from multiple models.

Responsibilities:
- Combine predictions from multiple models
- Implement various ensemble strategies (voting, averaging, stacking)
- Track ensemble composition and performance
- Provide confidence aggregation across ensemble members
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    VOTING = "voting"  # Hard voting for classification
    AVERAGING = "averaging"  # Simple average for regression/probabilities
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average
    STACKING = "stacking"  # Meta-learner based
    SOFT_VOTING = "soft_voting"  # Probability-based voting


class VotingMethod(Enum):
    """Hard voting methods for classification."""
    MAJORITY = "majority"  # Most common prediction
    WEIGHTED_MAJORITY = "weighted_majority"  # Weighted by confidence


@dataclass
class EnsembleMember:
    """Represents a single model in an ensemble."""
    model_id: str
    weight: float = 1.0
    enabled: bool = True
    performance_score: float = 0.5  # Normalized 0-1 score for weighting
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'weight': self.weight,
            'enabled': self.enabled,
            'performance_score': self.performance_score
        }


@dataclass
class EnsemblePrediction:
    """Result of an ensemble prediction."""
    ensemble_id: str
    prediction: Union[float, int, np.ndarray]  # Primary prediction
    confidence: float  # 0-1 confidence score
    
    # Detailed information
    member_predictions: List[Any] = field(default_factory=list)
    member_confidences: List[float] = field(default_factory=list)
    prediction_variance: float = 0.0  # Diversity metric
    is_consensus: bool = True  # All ensemble members agree
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy_used: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ensemble_id': self.ensemble_id,
            'prediction': self._serialize_prediction(self.prediction),
            'confidence': self.confidence,
            'member_predictions': [self._serialize_prediction(p) for p in self.member_predictions],
            'member_confidences': self.member_confidences,
            'prediction_variance': self.prediction_variance,
            'is_consensus': self.is_consensus,
            'timestamp': self.timestamp,
            'strategy_used': self.strategy_used,
        }
    
    @staticmethod
    def _serialize_prediction(pred):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(pred, np.ndarray):
            return pred.tolist()
        return pred


class EnsembleManager:
    """Manages ensemble models and predictions."""
    
    def __init__(self, model_registry: Any = None):
        """
        Initialize ensemble manager.
        
        Args:
            model_registry: Reference to ModelRegistry for retrieving models
        """
        self.model_registry = model_registry
        self.ensembles: Dict[str, List[EnsembleMember]] = {}
        self.ensemble_configs: Dict[str, Dict[str, Any]] = {}
    
    def create_ensemble(
        self,
        ensemble_id: str,
        model_ids: List[str],
        strategy: EnsembleStrategy = EnsembleStrategy.AVERAGING,
        weights: Optional[List[float]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new ensemble configuration.
        
        Args:
            ensemble_id: Unique identifier for the ensemble
            model_ids: List of model IDs to include
            strategy: Ensemble strategy to use
            weights: Optional weights for ensemble members (auto-normalized)
            description: Description of the ensemble
            
        Returns:
            Ensemble configuration
        """
        if ensemble_id in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} already exists")
        
        if weights is None:
            weights = [1.0] * len(model_ids)
        
        if len(weights) != len(model_ids):
            raise ValueError("Weights length must match number of models")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Create ensemble members
        members = [
            EnsembleMember(model_id=mid, weight=w)
            for mid, w in zip(model_ids, normalized_weights)
        ]
        
        self.ensembles[ensemble_id] = members
        self.ensemble_configs[ensemble_id] = {
            'strategy': strategy.value,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'member_count': len(members),
        }
        
        logger.info(f"Created ensemble {ensemble_id} with {len(members)} members")
        return self.get_ensemble_info(ensemble_id)
    
    def get_ensemble_info(self, ensemble_id: str) -> Dict[str, Any]:
        """Get information about an ensemble."""
        if ensemble_id not in self.ensembles:
            raise KeyError(f"Ensemble {ensemble_id} not found")
        
        members = self.ensembles[ensemble_id]
        config = self.ensemble_configs[ensemble_id]
        
        return {
            'ensemble_id': ensemble_id,
            'members': [m.to_dict() for m in members],
            'config': config,
        }
    
    def predict(
        self,
        ensemble_id: str,
        X: Union[np.ndarray, List],
        strategy: Optional[EnsembleStrategy] = None,
    ) -> EnsemblePrediction:
        """
        Make prediction using ensemble.
        
        Args:
            ensemble_id: Ensemble to use for prediction
            X: Input features (single sample or batch)
            strategy: Override default strategy for this prediction
            
        Returns:
            EnsemblePrediction with ensemble result
        """
        if ensemble_id not in self.ensembles:
            raise KeyError(f"Ensemble {ensemble_id} not found")
        
        members = self.ensembles[ensemble_id]
        config = self.ensemble_configs[ensemble_id]
        
        if strategy is None:
            strategy = EnsembleStrategy(config['strategy'])
        
        # Collect predictions from enabled members
        member_predictions = []
        member_confidences = []
        
        for member in members:
            if not member.enabled:
                continue
            
            try:
                # Get model and predict
                if self.model_registry:
                    model, _ = self.model_registry.get_model(member.model_id)
                    member_pred = model.predict(X)
                    
                    # Try to get confidence
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        confidence = np.max(proba)
                    else:
                        confidence = member.performance_score
                else:
                    member_pred = None
                    confidence = 0.5
                
                if member_pred is not None:
                    member_predictions.append(member_pred)
                    member_confidences.append(confidence)
                    
            except Exception as e:
                logger.warning(f"Error predicting with model {member.model_id}: {e}")
        
        if not member_predictions:
            raise RuntimeError(f"No valid predictions from ensemble {ensemble_id}")
        
        # Combine predictions based on strategy
        if strategy == EnsembleStrategy.AVERAGING:
            final_pred = self._average_predictions(member_predictions)
            final_conf = np.mean(member_confidences)
        
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            final_pred = self._weighted_average_predictions(
                member_predictions,
                [m.weight for m in members if m.enabled]
            )
            final_conf = self._weighted_average(member_confidences, [m.weight for m in members if m.enabled])
        
        elif strategy == EnsembleStrategy.VOTING:
            final_pred = self._hard_voting(member_predictions)
            final_conf = self._voting_confidence(member_predictions)
        
        elif strategy == EnsembleStrategy.SOFT_VOTING:
            final_pred = self._soft_voting(member_confidences, member_predictions)
            final_conf = np.mean(member_confidences)
        
        else:
            final_pred = self._average_predictions(member_predictions)
            final_conf = np.mean(member_confidences)
        
        # Calculate prediction variance (diversity metric)
        pred_variance = np.var(member_predictions) if len(member_predictions) > 1 else 0.0
        
        # Check consensus
        is_consensus = len(set(str(p) for p in member_predictions)) == 1
        
        result = EnsemblePrediction(
            ensemble_id=ensemble_id,
            prediction=final_pred,
            confidence=float(final_conf),
            member_predictions=member_predictions,
            member_confidences=member_confidences,
            prediction_variance=float(pred_variance),
            is_consensus=is_consensus,
            strategy_used=strategy.value,
        )
        
        return result
    
    def batch_predict(
        self,
        ensemble_id: str,
        X: Union[np.ndarray, List],
        strategy: Optional[EnsembleStrategy] = None,
    ) -> List[EnsemblePrediction]:
        """
        Make predictions for a batch of samples.
        
        Args:
            ensemble_id: Ensemble to use
            X: Batch of input features
            strategy: Override default strategy
            
        Returns:
            List of EnsemblePredictions
        """
        results = []
        for sample in X:
            pred = self.predict(ensemble_id, sample.reshape(1, -1), strategy)
            results.append(pred)
        return results
    
    @staticmethod
    def _average_predictions(predictions: List[np.ndarray]) -> Union[float, np.ndarray]:
        """Simple averaging of predictions."""
        return np.mean(predictions, axis=0)
    
    @staticmethod
    def _weighted_average(values: List[float], weights: List[float]) -> float:
        """Weighted average of values."""
        total = sum(v * w for v, w in zip(values, weights))
        return total / sum(weights)
    
    @staticmethod
    def _weighted_average_predictions(
        predictions: List[np.ndarray],
        weights: List[float]
    ) -> Union[float, np.ndarray]:
        """Weighted average of predictions."""
        weighted_sum = np.sum([p * w for p, w in zip(predictions, weights)], axis=0)
        return weighted_sum / sum(weights)
    
    @staticmethod
    def _hard_voting(predictions: List[int]) -> int:
        """Hard voting for classification."""
        counts = {}
        for pred in predictions:
            counts[pred] = counts.get(pred, 0) + 1
        return max(counts, key=counts.get)
    
    @staticmethod
    def _soft_voting(confidences: List[float], predictions: List[Any]) -> Any:
        """Soft voting using confidence scores."""
        # Weight predictions by their confidence
        weighted_votes = {}
        for conf, pred in zip(confidences, predictions):
            weighted_votes[str(pred)] = weighted_votes.get(str(pred), 0) + conf
        
        best_pred = max(weighted_votes, key=weighted_votes.get)
        # Try to convert back to original type
        try:
            return int(best_pred)
        except:
            return best_pred
    
    @staticmethod
    def _voting_confidence(predictions: List[int]) -> float:
        """Calculate confidence from voting patterns."""
        total = len(predictions)
        if total == 0:
            return 0.0
        
        counts = {}
        for pred in predictions:
            counts[pred] = counts.get(pred, 0) + 1
        
        max_count = max(counts.values())
        return max_count / total
    
    def update_member_weight(self, ensemble_id: str, model_id: str, weight: float) -> None:
        """Update weight for a specific ensemble member."""
        if ensemble_id not in self.ensembles:
            raise KeyError(f"Ensemble {ensemble_id} not found")
        
        members = self.ensembles[ensemble_id]
        member = next((m for m in members if m.model_id == model_id), None)
        
        if member is None:
            raise ValueError(f"Model {model_id} not in ensemble {ensemble_id}")
        
        member.weight = weight
        logger.info(f"Updated weight for {model_id} in {ensemble_id} to {weight}")
    
    def disable_member(self, ensemble_id: str, model_id: str) -> None:
        """Disable a member of the ensemble."""
        if ensemble_id not in self.ensembles:
            raise KeyError(f"Ensemble {ensemble_id} not found")
        
        members = self.ensembles[ensemble_id]
        member = next((m for m in members if m.model_id == model_id), None)
        
        if member is None:
            raise ValueError(f"Model {model_id} not in ensemble {ensemble_id}")
        
        member.enabled = False
        logger.info(f"Disabled {model_id} in ensemble {ensemble_id}")
    
    def enable_member(self, ensemble_id: str, model_id: str) -> None:
        """Enable a member of the ensemble."""
        if ensemble_id not in self.ensembles:
            raise KeyError(f"Ensemble {ensemble_id} not found")
        
        members = self.ensembles[ensemble_id]
        member = next((m for m in members if m.model_id == model_id), None)
        
        if member is None:
            raise ValueError(f"Model {model_id} not in ensemble {ensemble_id}")
        
        member.enabled = True
        logger.info(f"Enabled {model_id} in ensemble {ensemble_id}")
    
    def list_ensembles(self) -> List[str]:
        """List all ensemble IDs."""
        return list(self.ensembles.keys())
