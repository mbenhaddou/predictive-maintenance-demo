# utils/config.py

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

@dataclass
class Config:
    OUTPUT_TYPE: str = 'binary'
    MODEL_TYPE: str = 'classification'
    MODEL_TYPE_OPTIONS: List[str] = field(default_factory=lambda: ['classification', 'regression'])
    OUTPUT_TYPES_OPTIONS: List[str] = field(default_factory=lambda: ['binary', 'multiclass', 'regression'])
    OUTPUT_COLUMN_OPTIONS: List[str] = field(default_factory=lambda: ['label_binary', 'label_multiclass', 'RUL'])
    OUTPUT_COLUMN: str = 'label_binary'
    DATASET_PATH: str = '../Dataset/'
    OUTPUT_PATH: str = '../Output/'
    ID_COLUMN: str = 'id'
    TIMESTEP_COLUMN: str = 'cycles'
    SEQUENCE_LENGTH: int = 30
    VALIDATION_SPLIT: float = 0.05
    RANDOM_STATE: int = 42
    W1: float = 10.0
    W0: float = 5.0
    LSTM_UNITS: List[int] = field(default_factory=lambda: [64, 32])
    DROPOUT_RATES: List[float] = field(default_factory=lambda: [0.2, 0.2])
    ACTIVATION: str = 'sigmoid'
    L2_REG: float = 0.001
    OPTIMIZER: str = 'adam'
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    BATCH_SIZE: int = 64
    SMOOTHING_FACTOR: float = 0.1
    STATUS_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'safe': '🟢',      # Safe
        'warning': '🟡',   # Warning
        'critical': '🔴'    # Critical
    })
    LABEL_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'safe': 'green',
        'warning': 'yellow',
        'critical': 'red'
    })
    class_labels: List[str] = field(default_factory=lambda: ['safe', 'warning', 'critical'])  # Example class labels
    BINARY_THRESHOLD: float = 0.5  # Threshold for binary classification
    REGRESSION_THRESHOLD: int = 20  # Example threshold for regression

    def update_from_dict(self, config_dict: Dict[str, any]):
        """Update configuration attributes from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, any]:
        """Convert configuration attributes to a dictionary."""
        return asdict(self)

    def save_to_file(self, config_file: str = 'config.json'):
        """Save the current configuration to a file."""
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Configuration saved to {config_file}.")

    def load_from_file(self, config_file: str = 'config.json'):
        """Load configuration attributes from a file."""
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            self.update_from_dict(config_data)
        print(f"Configuration loaded from {config_file}.")

    def get_model_path(self) -> str:
        """Construct the model path based on MODEL_TYPE and other configurations."""
        if self.OUTPUT_TYPE is None:
            raise ValueError("MODEL_TYPE is not set.")
        return os.path.join(self.OUTPUT_PATH, f'{self.OUTPUT_COLUMN}_model.weights.h5')
