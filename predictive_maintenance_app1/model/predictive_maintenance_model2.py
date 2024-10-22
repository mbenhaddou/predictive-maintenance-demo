"""
Predictive Maintenance using LSTM on NASA's Turbofan Engine Dataset

Supports:
- Binary Classification: Predict if an asset will fail within a certain time frame (e.g., cycles)
- Multiclass Classification: Categorize the failure severity or stages
- Regression: Predict the exact Remaining Useful Life (RUL)
"""

import os
import numpy as np
import logging
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, BatchNormalization, Masking
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
from predictive_maintenance_app.model.lstm_model import LSTMModel, LSTMConfig
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from predictive_maintenance_app.utils.config import Config


class PredictiveMaintenanceModel:
    """
    Class for building, training, evaluating, and saving the LSTM model.
    Supports binary classification, multiclass classification, and regression tasks.
    """

    def __init__(self, config: Config, nb_features: int, output_type: str):
        """
        Initializes the PredictiveMaintenanceModel with configuration parameters.

        Args:
            config (Config): Configuration instance containing model parameters.
            nb_features (int): Number of features in the input data.
            output_type (str): Type of output. Options: 'binary', 'multiclass', 'regression'.
        """
        self.output_type = output_type.lower()
        self.nb_features = nb_features
        self.config = config
        self.model = None

        # Configure output layer parameters based on task_type
        self._configure_output(config)

    def _configure_output(self, config: Config):
        """
        Configures output layer parameters, activation function, loss function, and metrics based on the output type.

        Args:
            config (Config): Configuration instance containing model parameters.
        """
        if self.output_type == 'binary':
            self.nb_out = 1
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        elif self.output_type == 'multiclass':
            self.nb_out = 3  # Adjust based on the number of classes
            self.activation = 'softmax'
            self.loss = 'sparse_categorical_crossentropy'  # Use 'categorical_crossentropy' if labels are one-hot encoded
            self.metrics = ['accuracy']
        elif self.output_type == 'regression':
            self.nb_out = 1
            self.activation = 'linear'
            self.loss = 'mean_squared_error'
            self.metrics = ['mae', 'mse']
        else:
            raise ValueError("Unsupported task_type. Choose from 'binary', 'multiclass', 'regression'.")

    def _build_model(self):
        config=LSTMConfig()
        config.USE_BATCH_NORMALIZATION=False
        config.USE_MASKING=True
        config.OUTPUT_COLUMN='RUL'
        config.LSTM_UNITS=[256]
        config.DROPOUT_RATES=[0.1]
        config.TASK_TYPE='regression'
        self.model = LSTMModel(config, self.nb_features)
        self.model.build_model()
        self.model.compile_model()

    def build_model(self, apply_masking: bool = True, batch_normalization: bool = False):
        """
        Builds and compiles the unified LSTM model architecture based on the configuration.
        Handles both classification and regression tasks.
        """
        self.model = Sequential()

        # Input Layer with Masking for Regression (if applicable)
        if apply_masking:
            self.model.add(Masking(mask_value=-99., input_shape=(self.config.SEQUENCE_LENGTH, self.nb_features)))

        # First LSTM layer
        self.model.add(LSTM(
            units=self.config.LSTM_UNITS[0],
            input_shape=(self.config.SEQUENCE_LENGTH, self.nb_features) if not apply_masking else None,
            return_sequences=True if len(self.config.LSTM_UNITS) > 1 else False,
            kernel_regularizer=l2(self.config.L2_REG)
        ))
        self.model.add(Dropout(self.config.DROPOUT_RATES[0]))
        if batch_normalization:
            self.model.add(BatchNormalization())

        # Additional LSTM layers (if any)
        for i in range(1, len(self.config.LSTM_UNITS)):
            return_seq = True if i < len(self.config.LSTM_UNITS) - 1 else False
            self.model.add(LSTM(
                units=self.config.LSTM_UNITS[i],
                return_sequences=return_seq,
                kernel_regularizer=l2(self.config.L2_REG)
            ))
            self.model.add(Dropout(self.config.DROPOUT_RATES[i]))
            self.model.add(BatchNormalization())

        # Output layer
        self.model.add(Dense(self.nb_out, activation=self.activation))

        # Compile model with specified optimizer and learning rate
        optimizer_instance = self._get_optimizer()
        self.model.compile(loss=self.loss, optimizer=optimizer_instance)

        logger.info("Model built and compiled successfully.")
        self.model.summary()


    def _get_optimizer(self):
        """
        Returns an optimizer instance based on the configuration.

        Returns:
            tf.keras.optimizers.Optimizer: Configured optimizer.
        """
        optimizer = self.config.OPTIMIZER.lower()
        learning_rate = self.config.LEARNING_RATE

        if optimizer == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            logger.warning("Unsupported optimizer. Defaulting to Adam.")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_model(self, seq_array: np.ndarray, label_array: np.ndarray, epochs: int, batch_size: int,
                   validation_split: float = 0.05, custom_callback: tf.keras.callbacks.Callback = None,
                   verbose: int = 1):
        """
        Trains the LSTM model with the provided data and configuration.

        Args:
            seq_array (np.ndarray): Training input sequences.
            label_array (np.ndarray): Training labels.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training batches.
            validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.05.
            custom_callback (tf.keras.callbacks.Callback, optional): Custom callback for training progress. Defaults to None.
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.

        Returns:
            tf.keras.callbacks.History: History object containing training metrics.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
            ModelCheckpoint(
                filepath=self.config.get_model_path(),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1,
                save_weights_only=True
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]

        # Add the custom callback if provided
        if custom_callback:
            callbacks.append(custom_callback)

        try:
            logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}.")
            history = self.model.train(
                seq_array, label_array,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
#                callbacks=callbacks
            )
            logger.info("Model training completed successfully.")
            return history
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def evaluate_model(self, seq_array: np.ndarray, label_array: np.ndarray, batch_size: int):
        """
        Evaluates the model on the provided data.

        Args:
            seq_array (np.ndarray): Input sequences for evaluation.
            label_array (np.ndarray): True labels for evaluation.
            batch_size (int): Batch size for evaluation.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            logger.info("Starting model evaluation.")

            # Perform evaluation
            # Use return_dict=True if supported to get a dictionary directly
            if hasattr(self.model, 'evaluate') and 'return_dict' in self.model.evaluate.__code__.co_varnames:
                scores = self.model.evaluate(
                    seq_array,
                    label_array,
                    verbose=1,
                    batch_size=batch_size,
                    return_dict=True
                )
            else:
                scores = self.model.evaluate(
                    seq_array,
                    label_array,
                    verbose=1,
                    batch_size=batch_size
                )

            evaluation_metrics = {}

            # Check if scores is a dictionary
            if isinstance(scores, dict):
                evaluation_metrics = scores
            # Check if scores is a list or tuple
            elif isinstance(scores, (list, tuple, np.ndarray)):
                # Retrieve metric names from the model
                metric_names = self.model.metrics_names  # e.g., ['loss', 'accuracy']
                for name, score in zip(metric_names, scores):
                    evaluation_metrics[name] = score
            # If scores is a scalar
            else:
                evaluation_metrics['loss'] = scores
                if self.metrics:
                    logger.warning("Model was evaluated without additional metrics.")

            # Additional metrics for regression
            if self.output_type == 'regression':
                y_pred = self.model.predict(seq_array, batch_size=batch_size)
                r2 = r2_score(label_array, y_pred)
                evaluation_metrics['r2_score'] = r2

            logger.info(f"Model evaluation completed with metrics: {evaluation_metrics}")
            return evaluation_metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def load_model_weights(self):
        """
        Loads model weights from the specified path. Assumes that the model architecture has already been built.
        """
        try:
            model_path = self.config.get_model_path()
            if os.path.isfile(model_path):
                self.model.load_weights(model_path)
                logger.info("Model weights loaded successfully.")
            else:
                logger.error(f"No saved model weights found at {model_path}. Please train the model first.")
                raise FileNotFoundError(f"No saved model weights found at {model_path}.")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise

    def load_and_build_model(self):
        """
        Builds the model architecture and loads weights from the specified path.
        """
        try:
            self.build_model()
            self.load_model_weights()
            logger.info("Model architecture built and weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error during model build/load: {e}")
            raise

    def predict(self, seq_array: np.ndarray):
        """
        Generates predictions for the provided sequences.

        Args:
            seq_array (np.ndarray): Input sequences for prediction.

        Returns:
            np.ndarray: Predicted classes or regression values.
        """
        try:
            logger.info("Generating predictions.")
            y_pred = self.model.predict(seq_array)
            if self.output_type == 'binary':
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
                return y_pred_class
            elif self.output_type == 'multiclass':
                y_pred_class = np.argmax(y_pred, axis=1)
                return y_pred_class
            elif self.output_type == 'regression':
                y_pred_reg = y_pred.flatten()
                return y_pred_reg
            else:
                raise ValueError("Unsupported task_type for prediction.")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save_full_model(self, save_path: str = None):
        """
        Saves the entire model (architecture + weights) to the specified path.

        Args:
            save_path (str, optional): Path to save the model. If None, saves to the default model_path with '.h5' extension.
        """
        try:
            if not save_path:
                save_path = os.path.splitext(self.config.get_model_path())[0] + '.h5'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Full model saved successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Error saving full model: {e}")
            raise

    def load_full_model(self, load_path: str = None):
        """
        Loads the entire model (architecture + weights) from the specified path.

        Args:
            load_path (str, optional): Path to load the model from. If None, loads from the default model_path with '.h5' extension.
        """
        try:
            if not load_path:
                load_path = os.path.splitext(self.config.get_model_path())[0] + '.h5'
            if os.path.isfile(load_path):
                self.model = tf.keras.models.load_model(load_path)
                logger.info(f"Full model loaded successfully from {load_path}.")
                self.model.summary()
            else:
                logger.error(f"No saved full model found at {load_path}. Please train and save the model first.")
                raise FileNotFoundError(f"No saved full model found at {load_path}.")
        except Exception as e:
            logger.error(f"Error loading full model: {e}")
            raise
