# model/lstm_model.py

import tensorflow as tf
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base_model import BaseModel
from predictive_maintenance_app.utils.config import Config

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class LSTMConfig(Config):
    # ---------------------------
    # 10. LSTM Configuration
    # ---------------------------
    LSTM_UNITS: List[int] = field(default_factory=lambda: [64, 32])
    DROPOUT_RATES: List[float] = field(default_factory=lambda: [0.2, 0.2])
    ACTIVATION: str = 'sigmoid'
    L2_REG: float = 0.001
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
    EPOCHS: int = 2
    BATCH_SIZE: int = 64
    SMOOTHING_FACTOR: float = 0.1

    def __post_init__(self):
        # Ensure that the number of LSTM units matches the number of dropout rates
        if len(self.LSTM_UNITS) != len(self.DROPOUT_RATES):
            raise ValueError("The number of LSTM_UNITS must match the number of DROPOUT_RATES.")


class LSTMModel(BaseModel):
    """
    Concrete implementation of BaseModel using LSTM architecture.
    """

    def __init__(self, config: LSTMConfig, nb_features: int):
        super().__init__(config, nb_features)

    def build_model(self):
        """
        Builds the LSTM model architecture based on the provided input shape.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (sequence_length, nb_features).
        """
        if not isinstance(self.config, LSTMConfig):
            raise TypeError("config must be an instance of LSTMConfig.")

        self.model = tf.keras.Sequential()
        input_shape = (self.config.SEQUENCE_LENGTH, self.nb_features)
        # Masking layer if needed (e.g., for padding)
        if self.config.USE_MASKING:
            self.model.add(tf.keras.layers.Masking(mask_value=-99., input_shape=input_shape))
            logger.info("Masking layer added.")

        # First LSTM layer
        self.model.add(tf.keras.layers.LSTM(
            units=self.config.LSTM_UNITS[0],
            return_sequences=True if len(self.config.LSTM_UNITS) > 1 else False,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.L2_REG),
            activation='tanh'  # Default LSTM activation
        ))
        self.model.add(tf.keras.layers.Dropout(self.config.DROPOUT_RATES[0]))
        if self.config.USE_BATCH_NORMALIZATION:
            self.model.add(tf.keras.layers.BatchNormalization())
            logger.info("BatchNormalization layer added.")

        # Additional LSTM layers if specified
        for i in range(1, len(self.config.LSTM_UNITS)):
            self.model.add(tf.keras.layers.LSTM(
                units=self.config.LSTM_UNITS[i],
                return_sequences=True if i < len(self.config.LSTM_UNITS) - 1 else False,
                kernel_regularizer=tf.keras.regularizers.l2(self.config.L2_REG),
                activation='tanh'
            ))
            self.model.add(tf.keras.layers.Dropout(self.config.DROPOUT_RATES[i]))
            if self.config.USE_BATCH_NORMALIZATION:
                self.model.add(tf.keras.layers.BatchNormalization())
                logger.info("BatchNormalization layer added.")

        # Output layer
        self.model.add(tf.keras.layers.Dense(self.nb_out, activation=self.activation))
        self.model.summary()
        logger.info("LSTM Model architecture built successfully.")

    def compile_model(self):
        """
        Compiles the LSTM model with the specified loss, optimizer, and metrics.
        """
        super().compile_model()
