import seaborn as sns; sns.set()
from predictive_maintenance_app.model.predictive_maintenance_model2 import PredictiveMaintenanceModel
import logging
logger = logging.getLogger(__name__)
# Assuming the PredictiveMaintenanceModel class is defined as above
import numpy as np
import random
import seaborn as sns
import tensorflow as tf

# Set Seaborn style
sns.set()

# Import custom modules
from predictive_maintenance_app.utils.config import Config
from predictive_maintenance_app.model.sequence_generator import SequenceGenerator


# Assuming the PredictiveMaintenanceModel class is defined as above

def main():
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Define configuration using the Config dataclass
    config = Config()

    # Load data using the DataLoader
    data_loader = load_data(config)

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

    config.EPOCHS = 20
    config.BATCH_SIZE = 128
    config.LSTM_UNITS=[256]
    config.OUTPUT_COLUMN='RUL'
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

    # Initialize the PredictiveMaintenanceModel for regression
    model = PredictiveMaintenanceModel(
        config=config,
        nb_features=train_array.shape[2],
        output_type='regression'
    )

    # Build the model
    model.build_model()

    # Convert data to float32 for TensorFlow
    train_array = np.array(train_array).astype(np.float32)
    label_array = np.array(label_array).astype(np.float32)
    test_array = np.array(test_array).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # Train the model
    logger.info("Starting model training...")
    history = model.train_model(
        seq_array=train_array,
        label_array=label_array,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model on training data
    logger.info("Evaluating model on training data...")
    evaluation_metrics_train = model.evaluate_model(
        seq_array=train_array,
        label_array=label_array,
        batch_size=config.BATCH_SIZE
    )
    print("Training Evaluation Metrics:", evaluation_metrics_train)

    # Evaluate the model on test data
    logger.info("Evaluating model on test data...")
    evaluation_metrics_test = model.evaluate_model(
        seq_array=test_array,
        label_array=y_test,
        batch_size=config.BATCH_SIZE
    )
    print("Test Evaluation Metrics:", evaluation_metrics_test)

    # Make predictions on a subset of test data
    logger.info("Making predictions on test data...")
    predictions = model.predict(test_array[:10])
    print("Predictions on Test Data:", predictions)

    # Save the full model
    logger.info("Saving the full model...")
    model.save_full_model()

    # Load the full model (for demonstration purposes)
    logger.info("Loading the full model...")
    loaded_model = PredictiveMaintenanceModel(
        config=config,
        nb_features=train_array.shape[2],
        output_type='regression'
    )
    loaded_model.load_full_model()

    # Make predictions with the loaded model to verify
    logger.info("Making predictions with the loaded model...")
    loaded_predictions = loaded_model.predict(test_array[:10])
    print("Loaded Model Predictions on Test Data:", loaded_predictions)

def load_data(_config):
    """Load data using DataLoader."""
    from predictive_maintenance_app.model.data_loader import DataLoader  # Import here to avoid circular imports
    data_loader = DataLoader(_config)
    data_loader.read_data()
    data_loader.output_column = _config.OUTPUT_COLUMN
    return data_loader

if __name__ == "__main__":
    main()

