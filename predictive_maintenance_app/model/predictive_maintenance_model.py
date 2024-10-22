# model/predictive_maintenance_model.py

import os
import numpy as np
import logging
from .base_model import BaseModel
from utils.config import Config

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class PredictiveMaintenanceModel:
    """
    Class for building, training, evaluating, and saving predictive maintenance models.
    Supports classification and regression tasks.
    """

    def __init__(self, model: BaseModel, nb_features: int, config: Config):
        """
        Initializes the PredictiveMaintenanceModel with a specific model.

        Args:
            model (BaseModel): An instance of a class that inherits from BaseModel.
            nb_features (int): Number of features in the input data.
            config (Config): Configuration instance containing model parameters.
        """
        self.task_type = config.TASK_TYPE.lower()
        self.nb_features = nb_features
        self.model = model
        self.config = config

    def build_and_compile_model(self):
        """
        Builds and compiles the model architecture.

        Args:
            input_shape (tuple): Shape of the input data (sequence_length, nb_features).
        """
        self.model.build_model()
        self.model.compile_model()
        logger.info("Model architecture built and compiled successfully.")

    def train(self, seq_array, label_array, epochs=None, batch_size=None, validation_split=None, custom_callback=None):
        """
        Trains the underlying model.

        Args:
            seq_array (np.array): Training input sequences.
            label_array (np.array): Training labels.
            epochs (int, optional): Number of training epochs. If None, uses config's EPOCHS.
            batch_size (int, optional): Size of training batches. If None, uses config's BATCH_SIZE.
            validation_split (float, optional): Fraction of data to use for validation. If None, uses config's VALIDATION_SPLIT.
            custom_callback (list of tf.keras.callbacks.Callback, optional): Custom callbacks for training progress. Defaults to None.

        Returns:
            tf.keras.callbacks.History: History object containing training metrics.
        """
        epochs = epochs if epochs is not None else self.config.EPOCHS
        batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE
        validation_split = validation_split if validation_split is not None else self.config.VALIDATION_SPLIT

        history = self.model.train(
            np.array(seq_array).astype(np.float32),
            np.array(label_array).astype(np.float32),
            epochs,
            batch_size,
            validation_split,
            custom_callback
        )
        logger.info("Model training completed successfully.")
        return history

    def evaluate(self, seq_array, label_array, batch_size=None):
        """
        Evaluates the underlying model.

        Args:
            seq_array (np.array): Input sequences for evaluation.
            label_array (np.array): True labels for evaluation.
            batch_size (int, optional): Batch size for evaluation. If None, uses config's BATCH_SIZE.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE
        evaluation_metrics = self.model.evaluate(seq_array, label_array, batch_size)
        logger.info(f"Model evaluation metrics: {evaluation_metrics}")
        return evaluation_metrics

    def predict(self, seq_array):
        """
        Generates predictions using the underlying model.

        Args:
            seq_array (np.array): Input sequences for prediction.

        Returns:
            np.array: Predicted classes or regression values.
        """
        y_pred = self.model.predict(seq_array)
        logger.info("Model prediction completed.")
        return y_pred

    def save_weights(self, filepath=None):
        """
        Saves the underlying model's weights.

        Args:
            filepath (str, optional): Path to save the weights. If None, uses model's default path.
        """
        self.model.save_weights(filepath)
        logger.info("Model weights saved successfully.")


    def load_and_build_model(self):
        """
        Builds the model architecture and loads weights from the specified path.
        """
        try:
            self.model.load_and_build_model(self.config.get_model_path())
            logger.info("Model architecture built and weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error during model build/load: {e}")
            raise
    def load_weights(self, filepath=None):
        """
        Loads weights into the underlying model.

        Args:
            filepath (str, optional): Path to load the weights from. If None, uses model's default path.
        """
        self.model.load_weights(filepath)
        logger.info("Model weights loaded successfully.")

    def save_full_model(self, filepath=None):
        """
        Saves the entire underlying model (architecture + weights).

        Args:
            filepath (str, optional): Path to save the full model. If None, uses model's default path with '.h5' extension.
        """
        if filepath is None:
            filepath = self.config.get_model_path()
        self.model.save_full_model(filepath)
        logger.info("Full model saved successfully.")

    def load_full_model(self, load_path=None):
        """
        Loads the entire underlying model (architecture + weights).

        Args:
            load_path (str, optional): Path to load the full model from. If None, uses model's default path with '.h5' extension.
        """
        if load_path is None:
            load_path = self.config.get_model_path()
        self.model.load_full_model(load_path)
        logger.info("Full model loaded successfully.")
