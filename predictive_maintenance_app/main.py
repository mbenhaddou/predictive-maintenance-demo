# main.py

import copy
import logging
import random

import numpy as np
import seaborn as sns
import tensorflow as tf

from predictive_maintenance_app.model.data_loader import DataLoader
from predictive_maintenance_app.model.sequence_generator import SequenceGenerator
from predictive_maintenance_app.model.lstm_model import LSTMConfig
from predictive_maintenance_app.model.model_registry import MODEL_REGISTRY

# Set Seaborn style
sns.set()

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def load_data(_config: LSTMConfig) -> DataLoader:
    """Load data using DataLoader."""
    data_loader = DataLoader(_config)
    data_loader.read_data()
    data_loader.output_column = _config.OUTPUT_COLUMN
    return data_loader


def main():
    # Define base configuration using the LSTMConfig dataclass
    base_config = LSTMConfig()
    base_config.EPOCHS=2
    # Select the desired model from the registry
    desired_model_name = "LSTM Regression Model"
    if desired_model_name not in MODEL_REGISTRY:
        logger.error(f"Model '{desired_model_name}' not found in MODEL_REGISTRY.")
        raise ValueError(f"Model '{desired_model_name}' not found in MODEL_REGISTRY.")

    model_info = MODEL_REGISTRY[desired_model_name]
    logger.info(f"Selected Model: {desired_model_name}")

    # Create a deep copy of the base configuration to avoid modifying it
    config = copy.deepcopy(base_config)

    # Apply configuration overrides
    config.update_from_dict(model_info.get("config_overrides", {}))
    logger.info(f"Applied configuration overrides for '{desired_model_name}'.")
    config.OUTPUT_COLUMN = 'RUL'
    # Initialize the DataLoader with the updated configuration
    data_loader = load_data(config)
    config.MODEL_NAME = desired_model_name

    # Define sensors to keep and sensors to drop
    remaining_sensors = [
        's2', 's3', 's4', 's7', 's8', 's9',
        's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
    ]
    drop_sensors = [
        element for element in data_loader.train_df.columns
        if element not in remaining_sensors + [
            'id', 'cycle', 'voltage_input', 'current_limit',
            'speed_control', 'RUL', 'label_binary', 'label_multiclass'
        ]
    ]

    # Data Preparation
    logger.info("Preparing data...")
    data_loader.prepare_data(drop_sensors, remaining_sensors, config.SMOOTHING_FACTOR)

    # Generate sequences using SequenceGenerator
    sequence_cols = data_loader.get_sequence_cols()
    seq_gen = SequenceGenerator(
        df=data_loader.train_df,
        sequence_length=config.SEQUENCE_LENGTH,
        sequence_cols=sequence_cols,
        output_column=config.OUTPUT_COLUMN
    )
    train_array, label_array = seq_gen.generate_sequences(padding_strategy='zero')

    # Generate test sequences
    test_array, y_test = data_loader.generate_test_sequences(config.SEQUENCE_LENGTH, config.OUTPUT_COLUMN)

    logger.info(f"Training data shape: {train_array.shape}")
    logger.info(f"Training labels shape: {label_array.shape}")
    logger.info(f"Test data shape: {test_array.shape}")
    logger.info(f"Test labels shape: {y_test.shape}")

    config.USE_BATCH_NORMALIZATION = False
    # Initialize the model
    model_class = model_info["model_class"]
    model = model_class(
        config=config,
        nb_features=train_array.shape[2]
    )

    # Build and compile the model
    model.build_model()
    model.compile_model()

    # Convert data to float32 for TensorFlow
    train_array = np.array(train_array).astype(np.float32)
    label_array = np.array(label_array).astype(np.float32)
    test_array = np.array(test_array).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # Train the model
    logger.info("Starting model training...")
    history = model.train(
        seq_array=train_array,
        label_array=label_array,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model on training data
    logger.info("Evaluating model on training data...")
    evaluation_metrics_train = model.evaluate(
        seq_array=train_array,
        label_array=label_array,
        batch_size=config.BATCH_SIZE
    )
    print(f"[{desired_model_name}] Training Evaluation Metrics:", evaluation_metrics_train)

    # Evaluate the model on test data
    logger.info("Evaluating model on test data...")
    evaluation_metrics_test = model.evaluate(
        seq_array=test_array,
        label_array=y_test,
        batch_size=config.BATCH_SIZE
    )
    print(f"[{desired_model_name}] Test Evaluation Metrics:", evaluation_metrics_test)

    # Make predictions on a subset of test data
    logger.info("Making predictions on test data...")
    predictions = model.predict(test_array[:10])
    print(f"[{desired_model_name}] Predictions on Test Data:", predictions)

    # Save the full model
    logger.info("Saving the full model...")
    model.save_full_model()

    # Load the full model (for verification purposes)
    logger.info("Loading the full model for verification...")

    loaded_model = model_class(
        config=config,
        nb_features=train_array.shape[2]
    )
    loaded_model.load_full_model()

    # Make predictions with the loaded model to verify
    logger.info("Making predictions with the loaded model...")
    loaded_predictions = loaded_model.predict(test_array[:10])
    print(f"[{desired_model_name}] Loaded Model Predictions on Test Data:", loaded_predictions)

    logger.info(f"--- Completed Processing Model: {desired_model_name} ---\n")


if __name__ == "__main__":
    main()
