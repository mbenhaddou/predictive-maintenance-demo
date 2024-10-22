# model/base_model.py

from abc import ABC, abstractmethod
import logging
import tensorflow as tf
from typing import Optional, List, Dict, Any
from sklearn.metrics import r2_score
import os
import numpy as np

from utils.config import Config

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BaseModel(ABC):
    """
    Abstract Base Model class for building, training, evaluating, and saving models.
    Supports binary classification, multiclass classification, and regression tasks.
    """

    def __init__(self, config: Config, nb_features: int):
        """
        Initializes the BaseModel with configuration parameters.

        Args:
            config (Config): Configuration instance containing model parameters.
            nb_features (int): Number of features in the input data.
        """
        self.task_type = config.TASK_TYPE.lower()
        self.classification_type = config.CLASSIFICATION_TYPE.lower()
        self.nb_features = nb_features
        self.config = config
        self.model = None

        # Configure output layer parameters based on task_type
        self._configure_output()

    def _configure_output(self):
        """
        Configures output layer parameters, activation function, loss function, and metrics based on the output type.

        Args:
            config (Config): Configuration instance containing model parameters.
        """
        if self.task_type=='classification':
            if self.classification_type == 'binary':
                self.nb_out = 1
                self.activation = 'sigmoid'
                self.loss = 'binary_crossentropy'
                self.metrics = ['accuracy']
            elif self.classification_type == 'multiclass':
                self.nb_out = 3  # Adjust based on the number of classes
                self.activation = 'softmax'
                self.loss = 'sparse_categorical_crossentropy'  # Use 'categorical_crossentropy' if labels are one-hot encoded
                self.metrics = ['accuracy']
        elif self.task_type == 'regression':
            self.nb_out = 1
            self.activation = 'linear'
            self.loss = 'mean_squared_error'
            self.metrics = ['mae', 'mse']
        else:
            raise ValueError("Unsupported task_type. Choose from 'binary', 'multiclass', 'regression'.")

    @abstractmethod
    def build_model(self):
        """Builds the model architecture."""
        pass

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Retrieves the optimizer instance based on the configuration.

        Returns:
            tf.keras.optimizers.Optimizer: Optimizer instance.
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

    def compile_model(self):
        """
        Compiles the model with specified loss, optimizer, and metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        optimizer_instance = self._get_optimizer()
        self.model.compile(loss=self.loss, optimizer=optimizer_instance, metrics=self.metrics)
        logger.info("Model compiled successfully.")

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Parses the CALLBACKS_CONFIG from the configuration and instantiates the corresponding Keras callbacks.
        Replaces placeholders like '{model_path}' with actual values from the configuration.

        Returns:
            List[tf.keras.callbacks.Callback]: List of instantiated Keras callbacks.
        """
        callbacks = []

        for cb_config in self.config.CALLBACKS_CONFIG:
            cb_type = cb_config.get('type')
            if not cb_type:
                logger.error("Callback configuration missing 'type' key.")
                raise ValueError("Each callback configuration must have a 'type' key.")

            # Create a copy to avoid modifying the original config
            cb_params = cb_config.copy()

            # Handle placeholders
            if cb_type == "ModelCheckpoint":
                # Replace 'filepath' placeholder with actual model path
                filepath = cb_params.get('filepath')
                if filepath == "{filepath}":
                    cb_params['filepath'] = self.config.get_model_path()
                elif filepath is None:
                    raise ValueError("ModelCheckpoint callback requires 'filepath' parameter.")
                # Else, use the provided filepath as is

            # Remove the 'type' key before passing to the callback constructor
            cb_params.pop('type')

            try:
                # Dynamically get the callback class from keras.callbacks
                callback_class = getattr(tf.keras.callbacks, cb_type)
                # Instantiate the callback with parameters
                callback_instance = callback_class(**cb_params)
                callbacks.append(callback_instance)
                logger.info(f"Callback '{cb_type}' instantiated with parameters: {cb_params}")
            except AttributeError:
                logger.error(f"Callback type '{cb_type}' is not a valid Keras callback.")
                raise ValueError(f"Callback type '{cb_type}' is not a valid Keras callback.")
            except TypeError as e:
                logger.error(f"Error instantiating callback '{cb_type}': {e}")
                raise

        return callbacks

    def train(
        self,
        seq_array: np.ndarray,
        label_array: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_split: Optional[float] = None,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Trains the underlying model.

        Args:
            seq_array (np.ndarray): Training input sequences.
            label_array (np.ndarray): Training labels.
            epochs (int, optional): Number of training epochs. If None, uses config's EPOCHS.
            batch_size (int, optional): Size of training batches. If None, uses config's BATCH_SIZE.
            validation_split (float, optional): Fraction of data to use for validation. If None, uses config's VALIDATION_SPLIT.
            custom_callbacks (List[tf.keras.callbacks.Callback], optional): Custom callbacks for training progress.
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
            tf.keras.callbacks.History: History object containing training metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        epochs = epochs if epochs is not None else self.config.EPOCHS
        batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE
        validation_split = validation_split if validation_split is not None else self.config.VALIDATION_SPLIT

        # Instantiate callbacks from configuration
        config_callbacks = self.get_callbacks()

        # Combine with custom callbacks
        all_callbacks = config_callbacks
        if custom_callbacks:
            all_callbacks.extend(custom_callbacks)

        history = self.model.fit(
            seq_array,
            label_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=all_callbacks,
            verbose=verbose
        )
        logger.info("Model training completed successfully.")
        return history

    def evaluate(
        self,
        seq_array: np.ndarray,
        label_array: np.ndarray,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the underlying model.

        Args:
            seq_array (np.ndarray): Input sequences for evaluation.
            label_array (np.ndarray): True labels for evaluation.
            batch_size (int, optional): Batch size for evaluation. If None, uses config's BATCH_SIZE.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE

        scores = self.model.evaluate(seq_array, label_array, batch_size=batch_size, verbose=1)

        evaluation_metrics = {'loss': scores[0]}
        for idx, metric in enumerate(self.metrics):
            evaluation_metrics[metric] = scores[idx + 1]

        # Additional metrics for regression
        if self.task_type == 'regression':
            y_pred = self.model.predict(seq_array, batch_size=batch_size)
            r2 = r2_score(label_array, y_pred)
            evaluation_metrics['r2_score'] = r2

        logger.info(f"Model evaluation metrics: {evaluation_metrics}")
        return evaluation_metrics

    def predict(self, seq_array: np.ndarray) -> np.ndarray:
        """
        Generates predictions using the underlying model.

        Args:
            seq_array (np.ndarray): Input sequences for prediction.

        Returns:
            np.ndarray: Predicted classes or regression values.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        y_pred = self.model.predict(seq_array)
        logger.info("Model prediction completed.")

        if self.task_type == 'binary':
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
            return y_pred_class
        elif self.task_type == 'multiclass':
            y_pred_class = np.argmax(y_pred, axis=1)
            return y_pred_class
        elif self.task_type == 'regression':
            y_pred_reg = y_pred.flatten()
            return y_pred_reg
        else:
            raise ValueError("Unsupported task_type for prediction.")

    def save_weights(self, filepath: Optional[str] = None):
        """
        Saves the underlying model's weights.

        Args:
            filepath (str, optional): Path to save the weights. If None, uses model's default path.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        filepath = filepath if filepath is not None else self.config.get_model_path()
        self.model.save_weights(filepath)
        logger.info(f"Model weights saved successfully at {filepath}.")

    def load_weights(self, filepath: Optional[str] = None):
        """
        Loads weights into the underlying model.

        Args:
            filepath (str, optional): Path to load the weights from. If None, uses model's default path.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        filepath = filepath if filepath is not None else self.config.get_model_path()
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No weights file found at {filepath}.")
        self.model.load_weights(filepath)
        logger.info(f"Model weights loaded successfully from {filepath}.")

    def save_full_model(self, filepath: Optional[str] = None):
        """
        Saves the entire underlying model (architecture + weights).

        Args:
            filepath (str, optional): Path to save the full model. If None, uses model's default path with '.h5' extension.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        if filepath is None:
            base, _ = os.path.splitext(self.config.get_model_path())
            filepath = f"{base}.h5"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Full model saved successfully at {filepath}.")

    def load_full_model(self, filepath: Optional[str] = None):
        """
        Loads the entire underlying model (architecture + weights).

        Args:
            filepath (str, optional): Path to load the model from. If None, uses model's default path with '.h5' extension.
        """
        if filepath is None:
            base, _ = os.path.splitext(self.config.get_model_path())
            filepath = f"{base}.h5"

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}.")

        self.model = tf.keras.models.load_model(filepath)
        self._configure_output()  # Reconfigure output parameters based on the loaded model
        logger.info(f"Full model loaded successfully from {filepath}.")
        self.model.summary()

    def load_and_build_model(self, filepath):
        """
        Builds the model architecture and loads weights from the specified path.
        """
        try:
            self.build_model()
            self.load_weights(filepath)
            logger.info("Model architecture built and weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error during model build/load: {e}")
            raise