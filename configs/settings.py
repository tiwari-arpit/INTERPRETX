"""
Configuration settings for INTERPRETX.

Manages all environment variables, paths, and configuration parameters.
"""

import os
from pathlib import Path
from typing import Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
STORAGE_PATH = PROJECT_ROOT / "storage"
MODELS_PATH = STORAGE_PATH / "models"
LOGS_PATH = STORAGE_PATH / "logs"

# Ensure storage directories exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Model Registry Configuration
MODEL_REGISTRY_PATH = str(MODELS_PATH / "registry")
MODEL_METADATA_FILE = str(MODELS_PATH / "registry" / "metadata.json")

# Prediction Engine Configuration
ENABLE_UNCERTAINTY = True
MAX_PREDICTION_HISTORY = 10000
PREDICTION_LOG_FILE = str(LOGS_PATH / "predictions.log")

# Ensemble Configuration
ENABLE_ENSEMBLE = True
MAX_ENSEMBLE_SIZE = 10

# Monitoring Configuration
ENABLE_DRIFT_MONITORING = True
DRIFT_LOG_FILE = str(LOGS_PATH / "drift_detection.log")

# Feature Engineering
DEFAULT_FEATURE_SCALING = "standard"  # 'standard', 'minmax', 'robust'
HANDLE_MISSING_VALUES = True

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = str(LOGS_PATH / "app.log")

# Database Configuration (optional for future use)
DATABASE_URL = os.getenv("DATABASE_URL", None)
USE_DATABASE = False

# API Configuration (for FastAPI deployment)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# UI Configuration (Streamlit)
STREAMLIT_PAGE_TITLE = "INTERPRETX - Decision Intelligence Platform"
STREAMLIT_PAGE_ICON = "🤖"
STREAMLIT_LAYOUT = "wide"

# Model Performance Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_MEDIUM = 0.65
CONFIDENCE_THRESHOLD_LOW = 0.50

# Risk Scoring Thresholds
RISK_THRESHOLD_GREEN = 0.3   # Safe to auto-approve
RISK_THRESHOLD_YELLOW = 0.7  # Review recommended
RISK_THRESHOLD_RED = 0.9     # Escalate to human

# Feature Validation
MAX_FEATURE_DIMENSIONS = 1000
MIN_SAMPLE_SIZE = 1
MAX_BATCH_SIZE = 10000

# Memory and Performance
CACHE_PREDICTIONS = True
MAX_CACHED_PREDICTIONS = 1000
PREDICTION_TIMEOUT_SECONDS = 30

# Data Drift Configuration
DRIFT_DETECTION_ENABLED = True
DRIFT_WINDOW_SIZE = 100  # Check drift over last N predictions
DRIFT_THRESHOLD = 0.05   # % change threshold

# Uncertainty Estimation
UNCERTAINTY_METHOD = "confidence"  # 'confidence', 'entropy', 'ensemble_std'
CONFIDENCE_INTERVAL_ALPHA = 0.95   # 95% CI

def get_storage_path(subfolder: Optional[str] = None) -> Path:
    """
    Get storage path for a specific component.
    
    Args:
        subfolder: Optional subfolder name
        
    Returns:
        Path to storage location
    """
    if subfolder:
        path = STORAGE_PATH / subfolder
    else:
        path = STORAGE_PATH
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_file(component: str) -> str:
    """
    Get log file path for a specific component.
    
    Args:
        component: Component name (e.g., 'predictions', 'drift', 'governance')
        
    Returns:
        Path to log file
    """
    log_file = LOGS_PATH / f"{component}.log"
    return str(log_file)


def validate_config() -> bool:
    """
    Validate configuration and storage setup.
    
    Returns:
        True if all checks pass
    """
    checks = [
        (MODELS_PATH.exists(), f"Models path exists: {MODELS_PATH}"),
        (LOGS_PATH.exists(), f"Logs path exists: {LOGS_PATH}"),
        (STORAGE_PATH.is_dir(), f"Storage is a directory: {STORAGE_PATH}"),
    ]
    
    all_valid = True
    for check, message in checks:
        if check:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            all_valid = False
    
    return all_valid
