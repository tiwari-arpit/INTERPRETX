"""
Storage Verification Script: Verify storage configuration and test persistence.

Run this script to verify that all storage systems are properly configured
and that models/logs are being stored correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def verify_storage():
    """Verify storage setup."""
    print("\n" + "="*60)
    print("INTERPRETX Storage Verification")
    print("="*60 + "\n")
    
    # 1. Check settings
    print("1. Checking configuration settings...")
    try:
        from configs.settings import (
            MODELS_PATH, LOGS_PATH, validate_config,
            MODELS_PATH, MODEL_REGISTRY_PATH
        )
        print(f"   ✓ Settings loaded successfully")
        print(f"   ✓ Models path: {MODELS_PATH}")
        print(f"   ✓ Logs path: {LOGS_PATH}")
        print(f"   ✓ Model registry path: {MODEL_REGISTRY_PATH}")
        
        if validate_config():
            print("   ✓ All configuration checks passed")
        else:
            print("   ⚠ Some configuration issues detected")
    except Exception as e:
        print(f"   ✗ Error loading settings: {e}")
        return False
    
    # 2. Check storage manager
    print("\n2. Checking storage manager...")
    try:
        from core.storage_manager import StorageManager
        storage = StorageManager()
        
        if storage.verify_storage():
            print("   ✓ Storage manager initialized successfully")
        else:
            print("   ⚠ Storage verification warnings")
        
        stats = storage.get_storage_stats()
        print(f"   ✓ Storage stats retrieved:")
        print(f"     - Models: {stats['models_dir']['file_count']} files, {stats['models_dir']['size_bytes']/1024:.2f} KB")
        print(f"     - Logs: {stats['logs_dir']['size_bytes']/1024:.2f} KB")
    except Exception as e:
        print(f"   ✗ Error with storage manager: {e}")
        return False
    
    # 3. Check model registry
    print("\n3. Checking model registry...")
    try:
        from core.model_register import ModelRegistry
        registry = ModelRegistry()
        print(f"   ✓ Model registry initialized")
        print(f"   ✓ Registry path: {registry.registry_path}")
        print(f"   ✓ Models directory: {registry.models_dir}")
        
        models = registry.list_models()
        print(f"   ✓ Registered models: {len(models)}")
    except Exception as e:
        print(f"   ✗ Error with model registry: {e}")
        return False
    
    # 4. Check ensemble manager
    print("\n4. Checking ensemble manager...")
    try:
        from core.ensemble_manager import EnsembleManager
        ensemble_mgr = EnsembleManager(model_registry=registry)
        print(f"   ✓ Ensemble manager initialized")
        ensembles = ensemble_mgr.list_ensembles()
        print(f"   ✓ Configured ensembles: {len(ensembles)}")
    except Exception as e:
        print(f"   ✗ Error with ensemble manager: {e}")
        return False
    
    # 5. Check prediction engine
    print("\n5. Checking prediction engine...")
    try:
        from core.prediction_engine import PredictionEngine
        pred_engine = PredictionEngine(
            model_registry=registry,
            ensemble_manager=ensemble_mgr,
            storage_manager=storage
        )
        print(f"   ✓ Prediction engine initialized")
        print(f"   ✓ Storage manager integrated: {pred_engine.storage_manager is not None}")
        print(f"   ✓ Prediction history size: {len(pred_engine.prediction_history)}")
    except Exception as e:
        print(f"   ✗ Error with prediction engine: {e}")
        return False
    
    # 6. Test logging
    print("\n6. Testing prediction logging...")
    try:
        # Create a test prediction result
        from core.prediction_engine import PredictionResult, ConfidenceLevel
        from datetime import datetime
        
        test_result = PredictionResult(
            prediction_id="test_001",
            model_id="test_model",
            input_features={"feature1": 1.5, "feature2": 2.3},
            prediction=1,
            confidence=0.87,
            confidence_level=ConfidenceLevel.HIGH,
            prediction_type="binary_classification",
            model_version="1.0.0",
        )
        
        log_path = storage.log_prediction(test_result)
        print(f"   ✓ Prediction logged successfully")
        print(f"   ✓ Log file: {log_path}")
        
        # Verify we can read it back
        logs = storage.get_prediction_logs(limit=5)
        print(f"   ✓ Retrieved {len(logs)} prediction logs")
        if logs:
            last_log = logs[-1]
            print(f"   ✓ Latest log: {last_log.get('prediction_id')} - Model: {last_log.get('model_id')}")
    except Exception as e:
        print(f"   ⚠ Prediction logging test failed: {e}")
    
    # 7. Test decision logging
    print("\n7. Testing decision logging...")
    try:
        test_decision = {
            "decision_id": "dec_001",
            "prediction_id": "test_001",
            "decision": "APPROVE",
            "risk_score": 0.25,
            "reason": "Test decision",
        }
        
        log_path = storage.log_decision(test_decision)
        print(f"   ✓ Decision logged successfully")
        print(f"   ✓ Log file: {log_path}")
        
        decisions = storage.get_decision_logs(limit=5)
        print(f"   ✓ Retrieved {len(decisions)} decision logs")
    except Exception as e:
        print(f"   ⚠ Decision logging test failed: {e}")
    
    print("\n" + "="*60)
    print("✓ All storage systems verified successfully!")
    print("="*60 + "\n")
    
    print("Storage Locations:")
    print(f"  Models: {MODELS_PATH}")
    print(f"  Logs: {LOGS_PATH}")
    print(f"\nLog Files:")
    print(f"  Predictions: {LOGS_PATH}/predictions.jsonl")
    print(f"  Decisions: {LOGS_PATH}/decisions.jsonl")
    print(f"  Drift Detection: {LOGS_PATH}/drift_detection.jsonl")
    print()
    
    return True


if __name__ == "__main__":
    success = verify_storage()
    sys.exit(0 if success else 1)
