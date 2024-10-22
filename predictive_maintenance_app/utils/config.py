# utils/config.py

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import os


@dataclass
class Config:
    # ---------------------------
    # 1. Task Type Selection
    # ---------------------------
    TASK_TYPE: str = 'classification'  # 'classification' or 'regression'
    TASK_TYPE_OPTIONS: List[str] = field(default_factory=lambda: ['classification', 'regression'])

    # ---------------------------
    # 2. Classification Subtype
    # ---------------------------
    CLASSIFICATION_TYPE: Optional[str] = 'binary'  # 'binary' or 'multiclass'
    CLASSIFICATION_TYPE_OPTIONS: List[str] = field(default_factory=lambda: ['binary', 'multiclass'])

    # ---------------------------
    # 3. Output Column Selection
    # ---------------------------
    OUTPUT_COLUMN_OPTIONS: List[str] = field(default_factory=lambda: ['label_binary', 'label_multiclass', 'RUL'])
    OUTPUT_COLUMN: str = 'label_binary'
    MODEL_NAME: str = 'LSTM Binary Classification Model'
    # ---------------------------
    # 4. Data Paths
    # ---------------------------
    DATASET_PATH: str = '../Dataset/'
    OUTPUT_PATH: str = '../Output/'

    # ---------------------------
    # 5. Data Columns
    # ---------------------------
    ID_COLUMN: str = 'id'
    TIMESTEP_COLUMN: str = 'cycles'

    # ---------------------------
    # 6. Sequence Generation
    # ---------------------------
    SEQUENCE_LENGTH: int = 30
    VALIDATION_SPLIT: float = 0.05
    RANDOM_STATE: int = 42

    # ---------------------------
    # 7. Label Generation Thresholds
    # ---------------------------
    W1: float = 10.0
    W0: float = 5.0


    # ---------------------------
    # 8. Visualization and Labeling
    # ---------------------------
    STATUS_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'safe': '🟢',  # Safe
        'warning': '🟡',  # Warning
        'critical': '🔴'  # Critical
    })
    LABEL_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'safe': 'green',
        'warning': 'yellow',
        'critical': 'red'
    })
    class_labels: List[str] = field(default_factory=lambda: ['safe', 'warning', 'critical'])  # Example class labels

    #----------------------------
    # 9. Other configs
    #----------------------------
    BINARY_THRESHOLD: float = 0.5  # Threshold for binary classification
    REGRESSION_THRESHOLD: int = 20  # Example threshold for regression

    CALLBACKS_CONFIG: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "type": "EarlyStopping",
            "monitor": "val_loss",
            "patience": 10,
            "verbose": 1,
            "mode": "min",
            "restore_best_weights": True
        },
        {
            "type": "ModelCheckpoint",
            "load_path": "{model_path}",  # Placeholder to be replaced
            "monitor": "val_loss",
            "save_best_only": True,
            "verbose": 1,
            "save_weights_only": True
        },
        {
            "type": "ReduceLROnPlateau",
            "monitor": "val_loss",
            "factor": 0.2,
            "patience": 5,
            "min_lr": 1e-6,
            "verbose": 1
        }
    ])
    # ---------------------------
    # 1P. Configuration Methods
    # ---------------------------
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration attributes from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
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
        """Construct the model path based on TASK_TYPE and other configurations."""
        if self.TASK_TYPE is None:
            raise ValueError("TASK_TYPE is not set.")
        return os.path.join(self.OUTPUT_PATH, f'{self.OUTPUT_COLUMN}_model.weights.h5')
