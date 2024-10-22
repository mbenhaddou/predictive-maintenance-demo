# predictive_maintenance_app/model/cnn_model.py
from dataclasses import dataclass

from model.base_model import BaseModel  # Assuming BaseModel is defined appropriately
from utils.config import Config
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class CNNConfig(Config):
    # ---------------------------
    # 10. CNN Configuration
    # ---------------------------
    CNN_FILTERS: List[int] = field(default_factory=lambda: [32, 64])
    KERNEL_SIZES: List[int] = field(default_factory=lambda: [3, 3])
    POOL_SIZES: List[int] = field(default_factory=lambda: [2, 2])
    DROPOUT_RATES: List[float] = field(default_factory=lambda: [0.25, 0.25])
    USE_MASKING: bool = False
    USE_BATCH_NORMALIZATION: bool = False

    # ---------------------------
    # 11. Optimizer Configuration
    # ---------------------------
    OPTIMIZER: str = 'adam'
    LEARNING_RATE: float = 0.001

    # ---------------------------
    # 12. Training Parameters
    # ---------------------------
    EPOCHS: int = 100
    BATCH_SIZE: int = 64
    SMOOTHING_FACTOR: float = 0.1

    def __post_init__(self):
        # Ensure that the number of CNN filters, kernel sizes, pool sizes, and dropout rates match
        if not (len(self.CNN_FILTERS) == len(self.KERNEL_SIZES) == len(self.POOL_SIZES) == len(self.DROPOUT_RATES)):
            raise ValueError("The number of CNN_FILTERS, KERNEL_SIZES, POOL_SIZES, and DROPOUT_RATES must match.")


class CNNModel(BaseModel):
    def __init__(self, config: CNNConfig, nb_features: int):
        super().__init__(config, nb_features)

    def build_model(self, input_shape: tuple):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

        self.model = Sequential()

        # Example CNN architecture
        for idx, (filters, kernel_size, pool_size, dropout_rate) in enumerate(zip(
                self.config.CNN_FILTERS,
                self.config.KERNEL_SIZES,
                self.config.POOL_SIZES,
                self.config.DROPOUT_RATES
        )):
            self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                                  input_shape=input_shape if idx == 0 else None))
            self.model.add(MaxPooling1D(pool_size=pool_size))
            self.model.add(Dropout(dropout_rate))
            if self.config.USE_BATCH_NORMALIZATION:
                self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_out, activation=self.activation))
