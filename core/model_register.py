"""
Model Registry: Centralized management of ML models across INTERPRETX.

Responsibilities:
- Register, retrieve, and manage model metadata
- Track model versions and performance metrics
- Support model lifecycle (training, validation, deployment, retirement)
- Provide model introspection and schema validation
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import joblib
import numpy as np


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle states."""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    model_type: str  # e.g., 'xgboost', 'sklearn', 'neural_network'
    status: ModelStatus
    created_at: str
    updated_at: str
    
    # Model characteristics
    input_features: List[str]
    output_type: str  # 'binary_classification', 'multi_class', 'regression', etc.
    
    # Performance and metadata
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Governance
    author: Optional[str] = None
    description: Optional[str] = None
    framework: Optional[str] = None  # 'sklearn', 'xgboost', 'torch', etc.
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


class ModelRegistry:
    """Central registry for managing ML models."""
    
    def __init__(self, registry_path: str = "./models_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path where model metadata and models are stored
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "metadata.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # In-memory registry
        self._registry: Dict[str, ModelMetadata] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for model_id, metadata in data.items():
                        metadata['status'] = ModelStatus(metadata['status'])
                        self._registry[model_id] = ModelMetadata(**metadata)
                logger.info(f"Loaded {len(self._registry)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self) -> None:
        """Persist registry to disk."""
        try:
            data = {
                model_id: meta.to_dict() 
                for model_id, meta in self._registry.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        model_id: str,
        model: Any,
        name: str,
        version: str,
        model_type: str,
        input_features: List[str],
        output_type: str,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        framework: Optional[str] = None,
        status: ModelStatus = ModelStatus.DEVELOPMENT,
    ) -> ModelMetadata:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model: The actual model object
            name: Human-readable model name
            version: Version string (e.g., '1.0.0')
            model_type: Type of model
            input_features: List of input feature names
            output_type: Type of output (classification, regression, etc.)
            training_metrics: Performance metrics on training set
            validation_metrics: Performance metrics on validation set
            feature_importance: Feature importance scores
            author: Author/creator of the model
            description: Model description
            framework: ML framework used
            status: Initial model status
            
        Returns:
            ModelMetadata: Metadata of registered model
        """
        if model_id in self._registry:
            raise ValueError(f"Model {model_id} already registered")
        
        now = datetime.now().isoformat()
        
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            status=status,
            created_at=now,
            updated_at=now,
            input_features=input_features,
            output_type=output_type,
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            feature_importance=feature_importance,
            author=author,
            description=description,
            framework=framework,
        )
        
        # Save model to disk
        model_file = self.models_dir / f"{model_id}_{version}.pkl"
        try:
            joblib.dump(model, model_file)
            metadata.file_path = str(model_file)
            logger.info(f"Model {model_id} saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            raise
        
        # Register in memory
        self._registry[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Model {model_id} registered successfully")
        return metadata
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve a model and its metadata.
        
        Args:
            model_id: Model identifier
            version: Specific version (if None, returns latest)
            
        Returns:
            Tuple of (model object, metadata)
        """
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id} not found in registry")
        
        metadata = self._registry[model_id]
        
        if not metadata.file_path or not Path(metadata.file_path).exists():
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
        
        try:
            model = joblib.load(metadata.file_path)
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific model."""
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id} not found")
        return self._registry[model_id]
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        model_type: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List registered models with optional filtering.
        
        Args:
            status: Filter by model status
            model_type: Filter by model type
            
        Returns:
            List of ModelMetadata
        """
        models = list(self._registry.values())
        
        if status:
            models = [m for m in models if m.status == status]
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        return sorted(models, key=lambda m: m.updated_at, reverse=True)
    
    def update_status(self, model_id: str, new_status: ModelStatus) -> ModelMetadata:
        """Update model status."""
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id} not found")
        
        metadata = self._registry[model_id]
        metadata.status = new_status
        metadata.updated_at = datetime.now().isoformat()
        
        self._save_registry()
        logger.info(f"Model {model_id} status updated to {new_status.value}")
        
        return metadata
    
    def update_metrics(
        self,
        model_id: str,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
    ) -> ModelMetadata:
        """Update model performance metrics."""
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id} not found")
        
        metadata = self._registry[model_id]
        
        if training_metrics:
            metadata.training_metrics.update(training_metrics)
        if validation_metrics:
            metadata.validation_metrics.update(validation_metrics)
        
        metadata.updated_at = datetime.now().isoformat()
        
        self._save_registry()
        logger.info(f"Model {model_id} metrics updated")
        
        return metadata
    
    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models in production."""
        return self.list_models(status=ModelStatus.PRODUCTION)
    
    def get_model_by_type(self, model_type: str) -> List[ModelMetadata]:
        """Get all models of a specific type."""
        return self.list_models(model_type=model_type)
    
    def compare_models(self, model_ids: List[str], metric: str = "validation_metrics") -> Dict[str, Any]:
        """
        Compare multiple models on a specific metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare ('training_metrics' or 'validation_metrics')
            
        Returns:
            Comparison dictionary
        """
        comparison = {}
        for model_id in model_ids:
            if model_id not in self._registry:
                continue
            
            metadata = self._registry[model_id]
            metrics = getattr(metadata, metric, {})
            comparison[model_id] = {
                'name': metadata.name,
                'version': metadata.version,
                'status': metadata.status.value,
                'metrics': metrics
            }
        
        return comparison
    
    def retire_model(self, model_id: str) -> ModelMetadata:
        """Mark model as archived."""
        return self.update_status(model_id, ModelStatus.ARCHIVED)
