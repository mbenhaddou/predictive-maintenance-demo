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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    """
    Class for building, training, evaluating, and saving the LSTM model.
    Supports binary classification, multiclass classification, and regression tasks.
    """

    def __init__(self, config, nb_features, output_type):
        """
        Initializes the PredictiveMaintenanceModel with configuration parameters.

        Args:
            config (Config): Configuration instance containing model parameters.
            nb_features (int): Number of features in the input data.
            output_type (str): Type of output. Options: 'binary', 'multiclass', 'regression'.
        """
        self.output_type = output_type.lower()
        self.nb_features = nb_features
#        self.sequence_length = config.SEQUENCE_LENGTH
        self.config = config
        self.model = None

        # Configure output layer parameters based on task_type
        self._configure_output(config)

    def _configure_output(self, config):
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
            self.nb_out = 3
            self.activation = 'softmax'
            self.loss = 'sparse_categorical_crossentropy'  # Use 'categorical_crossentropy' if labels are one-hot encoded
            self.metrics = ['accuracy']
        elif self.output_type == 'regression':
            self.nb_out = 1
            self.activation = 'linear'
            self.loss = 'mse'
            self.metrics = ['mae', 'mse']
        else:
            raise ValueError("Unsupported task_type. Choose from 'binary', 'multiclass', 'regression'.")

    def build_classification_model(self):
        """
        Builds and compiles the LSTM model architecture based on the configuration.
        """
        self.model = Sequential()

        # First LSTM layer
        self.model.add(LSTM(
            units=self.config.LSTM_UNITS[0],
            input_shape=(self.config.SEQUENCE_LENGTH, self.nb_features),
            return_sequences=True,
            kernel_regularizer=l2(self.config.L2_REG)
        ))
        self.model.add(Dropout(self.config.DROPOUT_RATES[0]))
        self.model.add(BatchNormalization())

        # Second LSTM layer
        self.model.add(LSTM(
            units=self.config.LSTM_UNITS[1],
            return_sequences=False,
            kernel_regularizer=l2(self.config.L2_REG)
        ))
        self.model.add(Dropout(self.config.DROPOUT_RATES[1]))
        self.model.add(BatchNormalization())

        # Output layer
        self.model.add(Dense(self.nb_out, activation=self.activation))  # Dynamic activation based on task_type

        # Compile model with specified optimizer and learning rate
        optimizer_instance = self._get_optimizer()
        self.model.compile(loss=self.loss, optimizer=optimizer_instance, metrics=self.metrics)

        logger.info("Model built and compiled successfully.")
        self.model.summary()

    def build_regression_model(self, input_shape, nodes_per_layer, dropout, activation, weights_file):
        self.model = Sequential()
        self.model.add(Masking(mask_value=-99., input_shape=input_shape))
        if len(nodes_per_layer) <= 1:
            self.model.add(LSTM(nodes_per_layer[0], activation=activation))
            self.model.add(Dropout(dropout))
        else:
            self.model.add(LSTM(nodes_per_layer[0], activation=activation, return_sequences=True))
            self.model.add(Dropout(dropout))
            self.model.add(LSTM(nodes_per_layer[1], activation=activation))
            self.model.add(Dropout(dropout))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.save_weights(weights_file)
        logger.info("Model built and compiled successfully.")
        self.model.summary()

    def _get_optimizer(self):
        """
        Returns an optimizer instance based on the configuration.

        Returns:
            tf.keras.optimizers.Optimizer: Configured optimizer.
        """
        if self.config.OPTIMIZER.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.config.LEARNING_RATE)
        else:
            logger.warning("Unsupported optimizer. Defaulting to Adam.")
            return tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)

    def train_model(self, seq_array, label_array, epochs, batch_size, validation_split=0.05, custom_callback=None):
        """
        Trains the LSTM model with the provided data and configuration.

        Args:
            seq_array (np.array): Training input sequences.
            label_array (np.array): Training labels.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training batches.
            custom_callback (tf.keras.callbacks.Callback, optional): Custom callback for training progress. Defaults to None.

        Returns:
            tf.keras.callbacks.History: History object containing training metrics.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
            ModelCheckpoint(self.config.get_model_path(), monitor='val_loss', save_best_only=True, mode='min', verbose=1,
                            save_weights_only=True)
        ]

        # Add the custom callback if provided
        if custom_callback:
            callbacks.append(custom_callback)

        try:
            history = self.model.fit(
                seq_array, label_array,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,  # Suppress console output; StreamlitCallback will handle updates
                callbacks=callbacks
            )
            logger.info("Model training completed successfully.")
            return history
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def evaluate_model(self, seq_array, label_array, batch_size):
        """
        Evaluates the model on the provided data.

        Args:
            seq_array (np.array): Input sequences for evaluation.
            label_array (np.array): True labels for evaluation.
            batch_size (int): Batch size for evaluation.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            # Perform evaluation
            scores = self.model.evaluate(seq_array, label_array, verbose=1, batch_size=batch_size)

            # Prepare a dictionary of metric names to their corresponding scores
            evaluation_metrics = {}
            # The first element is always 'loss'
            evaluation_metrics['loss'] = scores[0]
            # The rest depend on the task_type
            for idx, metric in enumerate(self.metrics):
                # Keras returns metrics in the order they were added
                evaluation_metrics[metric] = scores[idx + 1]

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
            if os.path.isfile(self.config.get_model_path()):
                self.model.load_weights(self.config.get_model_path())
                logger.info("Model weights loaded successfully.")
            else:
                logger.error(f"No saved model weights found at {self.config.get_model_path()}. Please train the model first.")
                raise FileNotFoundError(f"No saved model weights found at {self.config.get_model_path()}.")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise

    def load_and_build_model(self):
        """
        Builds the model architecture and loads weights from the specified path.
        """
        try:
            self.build_classification_model()
            self.load_model_weights()
            logger.info("Model architecture built and weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error during model build/load: {e}")
            raise

    def predict(self, seq_array):
        """
        Generates predictions for the provided sequences.

        Args:
            seq_array (np.array): Input sequences for prediction.

        Returns:
            np.array: Predicted classes or regression values.
        """
        try:
            y_pred = self.model.predict(seq_array)
            if self.output_type == 'binary':
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
            elif self.output_type == 'multiclass':
                y_pred_class = np.argmax(y_pred, axis=1)
            elif self.output_type == 'regression':
                y_pred_class = y_pred.flatten()
            else:
                raise ValueError("Unsupported task_type for prediction.")
            return y_pred_class
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save_full_model(self, save_path=None):
        """
        Saves the entire model (architecture + weights) to the specified path.

        Args:
            save_path (str, optional): Path to save the model. If None, saves to the default model_path with '.h5' extension.
        """
        try:
            if not save_path:
                save_path = os.path.splitext(self.config.get_model_path())[0] + '.h5'
            self.model.save(save_path)
            logger.info(f"Full model saved successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Error saving full model: {e}")
            raise

    def load_full_model(self, load_path=None):
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
            else:
                logger.error(f"No saved full model found at {load_path}. Please train and save the model first.")
                raise FileNotFoundError(f"No saved full model found at {load_path}.")
        except Exception as e:
            logger.error(f"Error loading full model: {e}")
            raise
