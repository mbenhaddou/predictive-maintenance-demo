# utils/config.py

import json, os


class Config:
    def __init__(self):
        # Initialize with default values
        self.OUTPUT_TYPE = 'binary'
        self.OUTPUT_TYPES_OPTIONS = ['label_binary', 'label_multiclass', 'RUL']
        self.OUTPUT_COLUMN = 'label_binary'
        self.DATASET_PATH = '../Dataset/'
        self.OUTPUT_PATH = '../Output/'
        self.SEQUENCE_LENGTH = 50
        self.VALIDATION_SPLIT=0.05
        self.RANDOM_STATE=42
        self.W1 = 10
        self.W0 = 5
        self.LSTM_UNITS = [64, 32]
        self.DROPOUT_RATES = [0.2, 0.2]
        self.L2_REG = 0.001
        self.OPTIMIZER = 'adam'
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 100
        self.BATCH_SIZE = 64
        self.SMOOTHING_FACTOR=0.1
        self.STATUS_COLORS = {
            'safe': '🟢',  # Safe
            'warning': '🟡',  # Warning
            'critical': '🔴'  # Critical
        }
        self.class_labels = ['safe', 'warning', 'critical']  # Example class labels
        self.BINARY_THRESHOLD = 0.5  # Threshold for binary classification
        self.REGRESSION_THRESHOLD = 20  # Example threshold for regression

    def update_from_dict(self, config_dict):
        """Update configuration attributes from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        """Convert configuration attributes to a dictionary."""
        return self.__dict__

    def save_to_file(self, config_file='config.json'):
        """Save the current configuration to a file."""
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load_from_file(self, config_file='config.json'):
        """Load configuration attributes from a file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            self.update_from_dict(config_data)
    def get_model_path(self):
        if self.OUTPUT_TYPE is None:
            raise ValueError("OUTPUT_TYPE is not set.")
        return os.path.join(self.OUTPUT_PATH, f'{self.OUTPUT_COLUMN}_model.weights.h5')