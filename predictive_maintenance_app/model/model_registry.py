# model_registry.py

from typing import Dict, Any
from model.lstm_model import LSTMModel, LSTMConfig
from model.cnn_model import CNNModel, CNNConfig  # Assuming you have a CNNModel

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "LSTM Binary Classification Model": {
        "model_class": LSTMModel,
        "config_class": LSTMConfig,  # Direct reference to the class
        "config_overrides": {
            "TASK_TYPE": "classification",
            "CLASSIFICATION_TYPE": "binary",
            "OUTPUT_COLUMN": "label_binary",
            "LSTM_UNITS": [64, 32],
            "DROPOUT_RATES": [0.2, 0.2],
            "SEQUENCE_LENGTH": 30,
            "USE_BATCH_NORMALIZATION": True,
            "USE_MASKING": False
        }
    },
    "LSTM Multiclass Classification Model": {
        "model_class": LSTMModel,
        "config_class": LSTMConfig,
        "config_overrides": {
            "TASK_TYPE": "classification",
            "CLASSIFICATION_TYPE": "multiclass",
            "OUTPUT_COLUMN": "label_multiclass",
            "LSTM_UNITS": [64, 32],
            "DROPOUT_RATES": [0.2, 0.2],
            "SEQUENCE_LENGTH": 30,
            "USE_MASKING": False,
            "USE_BATCH_NORMALIZATION": True
            # ... other overrides specific to this model
        }
    },
    "LSTM Regression Model": {
        "model_class": LSTMModel,
        "config_class": LSTMConfig,
        "config_overrides": {
            "TASK_TYPE": "regression",
            "OUTPUT_COLUMN": "RUL",
            "LSTM_UNITS": [256],
            "DROPOUT_RATES": [0.1],
            "SEQUENCE_LENGTH": 30,
            "USE_BATCH_NORMALIZATION": False,
            "USE_MASKING": True
        }
    },
    "CNN Binary Classification Model": {
        "model_class": CNNModel,
        "config_class": CNNConfig,
        "config_overrides": {
            "TASK_TYPE": "classification",
            "CLASSIFICATION_TYPE": "binary",
            "OUTPUT_COLUMN": "label_binary",
            "CNN_FILTERS": [32, 64],
            "KERNEL_SIZES": [3, 3],
            "POOL_SIZES": [2, 2],
            "DROPOUT_RATES": [0.25, 0.25],
            "SEQUENCE_LENGTH": 30,
            "USE_BATCH_NORMALIZATION": True,
            "USE_MASKING": False
            # ... other overrides specific to this model
        }
    },
    "CNN Multiclass Classification Model": {
        "model_class": CNNModel,
        "config_class": CNNConfig,
        "config_overrides": {
            "TASK_TYPE": "classification",
            "CLASSIFICATION_TYPE": "multiclass",
            "OUTPUT_COLUMN": "label_multiclass",
            "CNN_FILTERS": [32, 64],
            "KERNEL_SIZES": [3, 3],
            "POOL_SIZES": [2, 2],
            "DROPOUT_RATES": [0.25, 0.25],
            "SEQUENCE_LENGTH": 30,
            "USE_BATCH_NORMALIZATION": True,
            "USE_MASKING": False
            # ... other overrides specific to this model
        }
    },
    "CNN Regression Model": {
        "model_class": CNNModel,
        "config_class": CNNConfig,
        "config_overrides": {
            "TASK_TYPE": "regression",
            "OUTPUT_COLUMN": "RUL",
            "CNN_FILTERS": [64, 128],
            "KERNEL_SIZES": [3, 3],
            "POOL_SIZES": [2, 2],
            "DROPOUT_RATES": [0.3, 0.3],
            "SEQUENCE_LENGTH": 50,
            "USE_BATCH_NORMALIZATION": False,
            "USE_MASKING": True
            # ... other overrides specific to this model
        }
    }
    # Add more models here as needed
}
