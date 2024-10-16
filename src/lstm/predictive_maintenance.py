"""
Predictive Maintenance using LSTM on NASA's Turbofan Engine Dataset

Binary classification: Predict if an asset will fail within a certain time frame (e.g., cycles)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

##################################
# Configuration
##################################

class Config:
    DATASET_PATH = '../../Dataset/'
    OUTPUT_PATH = '../../Output/'
    SEQUENCE_LENGTH = 50
    W1 = 30  # Threshold for label generation
    W0 = 15  # Threshold for label generation
    LSTM_UNITS = [128, 64]  # LSTM units for each layer
    DROPOUT_RATES = [0.3, 0.3]  # Dropout rates for each layer
    L2_REG = 0.001  # L2 regularization factor
    EPOCHS = 50
    BATCH_SIZE = 256
    OPTIMIZER = 'adam'  # Optimizer
    LEARNING_RATE = 0.001  # Learning rate
    OUTPUT_COLUMN = "label_multiclass"  # Can be 'label_binary', 'label_multiclass', 'label_regression'
    OUTPUT_TYPE = None  # Will be set after detecting output type

    @classmethod
    def get_model_path(cls):
        if cls.OUTPUT_TYPE is None:
            raise ValueError("OUTPUT_TYPE is not set.")
        return os.path.join(cls.OUTPUT_PATH, f'{cls.OUTPUT_TYPE}_model.weights.h5')

# Create output directory if it doesn't exist
os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

##################################
# DataLoader Class
##################################

class DataLoader:
    """
    Class for loading and preprocessing data.
    """
    def __init__(self, dataset_path, sequence_length=50, w1=30, w0=15):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.w1 = w1
        self.w0 = w0
        self.scaler = MinMaxScaler()
        self.train_df = None
        self.test_df = None
        self.truth_df = None
        self.sequence_cols = None
        self.nb_features = None
        self.nb_out = 1
        self.output_column = None  # Initialize as None
        self.output_type = None  # Will be set dynamically


    def detect_output_type(self, df):
        """
        Detects the output type based on the output column values.
        """
        if pd.api.types.is_numeric_dtype(df[self.output_column]):
            unique_values = df[self.output_column].nunique()
            if unique_values == 2:
                self.output_type = 'binary'
            elif unique_values > 2 and unique_values < 10:  # Adjust threshold as needed
                self.output_type = 'multiclass'
            else:
                self.output_type = 'regression'
            Config.OUTPUT_TYPE = self.output_type
        else:
            raise ValueError(f"Output column '{self.output_column}' has non-numeric values, which is unsupported.")

    def read_data(self):
        """
        Reads and preprocesses the datasets.
        """
        try:
            self.train_df = self._read_file('PM_train.txt')
            self.test_df = self._read_file('PM_test.txt')
            self.truth_df = pd.read_csv(os.path.join(self.dataset_path, 'PM_truth.txt'), sep="\s+", header=None)
            self.truth_df.columns = ['RUL']
            self.train_df = self.train_df.sort_values(['id', 'cycle'])
            self._prepare_data()

            # Now that labels are generated and output_column is set, we can detect output type
            self.detect_output_type(self.train_df)

            logger.info("Data reading and preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error in reading data: {e}")
            raise

    def _read_file(self, filename):
        """
        Reads a data file and assigns column names.
        """
        df = pd.read_csv(os.path.join(self.dataset_path, filename), sep="\s+", header=None)
        df.columns = ['id', 'cycle',  'voltage_input', 'current_limit', 'speed_control'] + \
                     [f's{i}' for i in range(1, 22)]
        return df

    def _prepare_data(self):
        """
        Prepares the data by adding RUL, generating labels, and normalizing.
        """
        # Add RUL and labels to training data
        self.train_df = self._add_RUL(self.train_df)
        self.train_df = self._generate_labels(self.train_df)
        # Normalize training data
        self.train_df, self.scaler = self._normalize(self.train_df)
        # Normalize test data
        self.test_df, _ = self._normalize(self.test_df, self.scaler)
        # Prepare test data
        self.test_df = self._prepare_test_df()
        # Define sequence columns
        sensor_cols = [f's{i}' for i in range(1, 22)]
        self.sequence_cols = ['voltage_input', 'current_limit', 'speed_control', 'cycle'] + sensor_cols
        self.nb_features = len(self.sequence_cols)

    def _add_RUL(self, df):
        """
        Adds Remaining Useful Life (RUL) to the dataframe.
        """
        max_cycle = df.groupby('id')['cycle'].max().reset_index()
        max_cycle.columns = ['id', 'max_cycle']
        df = df.merge(max_cycle, on='id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df.drop('max_cycle', axis=1, inplace=True)
        return df

    def _generate_labels(self, df):
        """
        Generates labels based on the detected output type.
        """
        # For this example, let's generate all possible labels
        df['label_binary'] = np.where(df['RUL'] <= self.w1, 1, 0)
        df['label_multiclass'] = 0
        df.loc[df['RUL'] <= self.w1, 'label_multiclass'] = 1
        df.loc[df['RUL'] <= self.w0, 'label_multiclass'] = 2
        #df['label_regression'] = df['RUL']

        # Now set the output_column based on your task
        # For example, if you're doing binary classification:
        self.output_column = 'label_binary'

        # If you want to automatically detect the output type, you can set it dynamically
        # For now, we'll assume it's set in Config
        self.output_column = Config.OUTPUT_COLUMN

        return df

    def _normalize(self, df, scaler=None):
        """
        Normalizes the dataframe using MinMaxScaler.
        """
        cols_normalize = df.columns.difference(['id', 'cycle', 'RUL', 'label_binary', 'label_multiclass'])
        if scaler is None:
            scaler = MinMaxScaler()
            df[cols_normalize] = scaler.fit_transform(df[cols_normalize])
        else:
            df[cols_normalize] = scaler.transform(df[cols_normalize])
        return df, scaler

    def _prepare_test_df(self):
        """
        Prepares the test dataframe by adding RUL and labels.
        """
        max_cycle = self.test_df.groupby('id')['cycle'].max().reset_index()
        max_cycle.columns = ['id', 'max_cycle']
        self.truth_df['id'] = self.truth_df.index + 1+22510000
        self.truth_df['max_cycle'] = max_cycle['max_cycle'] + self.truth_df['RUL']
        test_df = self.test_df.merge(self.truth_df[['id', 'max_cycle']], on='id', how='left')
        test_df['RUL'] = test_df['max_cycle'] - test_df['cycle']
        test_df.drop('max_cycle', axis=1, inplace=True)
        test_df = self._generate_labels(test_df)
        return test_df

    def get_train_data(self):
        """
        Returns the training dataframe.
        """
        return self.train_df

    def get_test_data(self):
        """
        Returns the test dataframe.
        """
        return self.test_df

    def get_sequence_cols(self):
        """
        Returns the sequence columns.
        """
        return self.sequence_cols

    def get_nb_features(self):
        """
        Returns the number of features.
        """
        return self.nb_features

    def get_sequence_length(self):
        """
        Returns the sequence length.
        """
        return self.sequence_length

##################################
# SequenceGenerator Class
##################################

class SequenceGenerator:
    """
    Class for generating sequences and labels for LSTM input.
    """

    def __init__(self, df, sequence_length, sequence_cols, output_column):
        self.df = df
        self.sequence_length = sequence_length
        self.sequence_cols = sequence_cols
        self.output_column = output_column

    def generate_sequences(self):
        """
        Generates sequences and labels for LSTM input.
        """
        seq_array, label_array = [], []
        for id in self.df['id'].unique():
            id_df = self.df[self.df['id'] == id]
            seq_gen = self._gen_sequence(id_df)
            seq_array.extend(seq_gen)
            labels = self._gen_labels(id_df)
            label_array.extend(labels)
        return np.array(seq_array), np.array(label_array)

    def _gen_sequence(self, id_df, padding_strategy='zero'):
        """
        Generates sequences for a single engine (id), including partial sequences with padding.
        """
        data_array = id_df[self.sequence_cols].values
        num_elements = data_array.shape[0]
        sequences = []
        num_features = data_array.shape[1]
        padding_length = self.sequence_length - 1

        if padding_strategy == 'zero':
            padding = np.zeros((padding_length, num_features))
        elif padding_strategy == 'mean':
            padding_value = np.mean(data_array, axis=0) if num_elements > 0 else np.zeros(num_features)
            padding = np.tile(padding_value, (padding_length, 1))
        else:
            raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

        padded_data = np.vstack((padding, data_array))

        for i in range(num_elements):
            start = i
            stop = i + self.sequence_length
            sequence = padded_data[start:stop, :]
            sequences.append(sequence)

        return sequences

    def _gen_labels(self, id_df):
        """
        Generates labels for the sequences.
        """
        labels = id_df[self.output_column].values
        return labels


##################################
# PredictiveMaintenanceModel Class
##################################

class PredictiveMaintenanceModel:
    """
    Class for building, training, evaluating, and saving the LSTM model.
    """

    def __init__(self, config, nb_features, output_type):
        self.output_type = output_type
        self.nb_features = nb_features
        self.sequence_length = config.SEQUENCE_LENGTH
        self.nb_out = 1
        self.model_path = config.get_model_path()
        self.lstm_units = config.LSTM_UNITS
        self.dropout_rates = config.DROPOUT_RATES
        self.l2_reg = config.L2_REG
        self.optimizer = config.OPTIMIZER
        self.learning_rate = config.LEARNING_RATE
        self.model = None

        self._configure_output()

    def _configure_output(self):
        """
        Configures output layer, activation, and loss based on the output type.
        """
        if self.output_type == 'binary':
            self.nb_out = 1
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        elif self.output_type == 'multiclass':
            self.nb_out = 3  # Adjust based on actual classes
            self.activation = 'softmax'
            self.loss = 'sparse_categorical_crossentropy'
            self.metrics = ['accuracy']
        elif self.output_type == 'regression':
            self.nb_out = 1
            self.activation = 'linear'
            self.loss = 'mse'
            self.metrics = ['mae', 'mse']

    def build_model(self):
        """
        Builds the LSTM model architecture.
        """
        self.model = Sequential()
        # First LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[0],
            input_shape=(self.sequence_length, self.nb_features),
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg)
        ))
        self.model.add(Dropout(self.dropout_rates[0]))
        self.model.add(BatchNormalization())
        # Second LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[1],
            return_sequences=False,
            kernel_regularizer=l2(self.l2_reg)
        ))
        self.model.add(Dropout(self.dropout_rates[1]))
        self.model.add(BatchNormalization())
        # Output layer
        self.model.add(Dense(self.nb_out, activation=self.activation))  # Use dynamic activation
        # Compile model with specified optimizer and learning rate
        optimizer_instance = self._get_optimizer()
        self.model.compile(loss=self.loss, optimizer=optimizer_instance,
                           metrics=self.metrics)  # Use dynamic loss and metrics
        logger.info("Model built successfully.")
        self.model.summary()

    def _get_optimizer(self):
        """
        Returns an optimizer instance based on configuration.
        """
        if self.optimizer.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            logger.warning("Unsupported optimizer. Defaulting to Adam.")
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    # In the PredictiveMaintenanceModel class

    def train_model(self, seq_array, label_array, epochs, batch_size, custom_callback=None):
        """
        Trains the LSTM model.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1,
                            save_weights_only=True)]
        # Add the custom callback if provided
        if custom_callback:
            callbacks.append(custom_callback)
        try:
            history = self.model.fit(
                seq_array, label_array,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.05,
                verbose=0,  # Suppress console output
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
        """
        try:
            scores = self.model.evaluate(seq_array, label_array, verbose=1, batch_size=batch_size)
            logger.info(f"Model evaluation completed with loss: {scores[0]}, accuracy: {scores[1]}")
            return scores
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def load_model(self):
        """
        Loads a saved model from the specified path.
        """
        try:
            if os.path.isfile(self.model_path):
                self.build_model()
                self.model.load_weights(self.model_path)
                logger.info("Model loaded successfully.")
            else:
                logger.error("No saved model found. Please train the model first.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, seq_array):
        """
        Generates predictions for the provided sequences.
        """
        try:
            y_pred = self.model.predict(seq_array)
            if self.output_type == 'binary':
                y_pred_class = (y_pred > 0.5).astype(int)
            elif self.output_type == 'multiclass':
                y_pred_class = np.argmax(y_pred, axis=1)
            elif self.output_type == 'regression':
                y_pred_class = y_pred.flatten()
            return y_pred_class
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


##################################
# Helper Functions
##################################

def plot_history(history, metric, output_path):
    """
    Plots training history for a given metric and saves the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric.capitalize()}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, f'model_{metric}.png'))
    plt.close()
    logger.info(f"Plot for {metric} saved.")

def evaluate_performance(y_true, y_pred, output_type, dataset_type=''):
    """
    Calculates and prints confusion matrix and performance metrics.
    """
    if output_type == 'binary':
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logger.info(f'\nConfusion Matrix ({dataset_type} Data):\n{cm}')
        logger.info(f'{dataset_type} Precision: {precision:.2f}')
        logger.info(f'{dataset_type} Recall: {recall:.2f}')
        logger.info(f'{dataset_type} F1-score: {f1:.2f}')
        print(f'\nConfusion Matrix ({dataset_type} Data):\n{cm}')
        print(f'{dataset_type} Precision: {precision:.2f}')
        print(f'{dataset_type} Recall: {recall:.2f}')
        print(f'{dataset_type} F1-score: {f1:.2f}')
    elif output_type == 'multiclass':
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f'\nConfusion Matrix ({dataset_type} Data):\n{cm}')
        logger.info(f'{dataset_type} Accuracy: {accuracy:.2f}')
        print(f'\nConfusion Matrix ({dataset_type} Data):\n{cm}')
        print(f'{dataset_type} Accuracy: {accuracy:.2f}')
    elif output_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logger.info(f'{dataset_type} MSE: {mse:.2f}')
        logger.info(f'{dataset_type} MAE: {mae:.2f}')
        logger.info(f'{dataset_type} R2 Score: {r2:.2f}')
        print(f'{dataset_type} MSE: {mse:.2f}')
        print(f'{dataset_type} MAE: {mae:.2f}')
        print(f'{dataset_type} R2 Score: {r2:.2f}')


def generate_test_sequences(df, sequence_length, sequence_cols, output_column):
    """
    Generates sequences and labels for test data.
    """
    seq_array_test = []
    label_array_test = []
    for id in df['id'].unique():
        id_df = df[df['id'] == id]
        if len(id_df) >= sequence_length:
            seq = id_df[sequence_cols].values[-sequence_length:]
            seq_array_test.append(seq)
            label = id_df[output_column].values[-1]
            label_array_test.append(label)
    return np.array(seq_array_test), np.array(label_array_test)

def plot_predictions(y_true, y_pred, output_path):
    """
    Plots predicted vs actual labels and saves the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.title('Predicted vs Actual Labels (Test Data)')
    plt.ylabel('Label')
    plt.xlabel('Sample')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'model_prediction_test.png'))
    plt.close()
    logger.info("Prediction plot saved.")


def add_features(df_in, rolling_win_size):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.

    Args:
            df_in (dataframe)     : The input dataframe to be proccessed (training or test)
            rolling_win_size (int): The window size, number of cycles for applying the rolling function

    Reurns:
            dataframe: contains the input dataframe with additional rolling mean and std for each sensor

    """

    sensor_cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
                   's16', 's17', 's18', 's19', 's20', 's21']

    sensor_av_cols = [nm.replace('s', 'av') for nm in sensor_cols]
    sensor_sd_cols = [nm.replace('s', 'sd') for nm in sensor_cols]

    df_out = pd.DataFrame()

    ws = rolling_win_size

    # calculate rolling stats for each engine id

    for m_id in pd.unique(df_in.id):
        # get a subset for each engine sensors
        df_engine = df_in[df_in['id'] == m_id]
        df_sub = df_engine[sensor_cols]

        # get rolling mean for the subset
        av = df_sub.rolling(ws, min_periods=1).mean()
        av.columns = sensor_av_cols

        # get the rolling standard deviation for the subset
        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
        sd.columns = sensor_sd_cols

        # combine the two new subset dataframes columns to the engine subset
        new_ftrs = pd.concat([df_engine, av, sd], axis=1)

        # add the new features rows to the output dataframe
        df_out = pd.concat([df_out, new_ftrs])

    return df_out

def prepare_train_data(df_in, period):
    """Add regression and classification labels to the training data.

        Regression label: ttf (time-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine
        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)
        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

      Args:
          df_in (dataframe): The input training data
          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

      Returns:
          dataframe: The input dataframe with regression and classification labels added

    """

    # create regression label

    # make a dataframe to hold the last cycle for each enginge in the dataset
    df_max_cycle = pd.DataFrame(df_in.groupby('id')['cycle'].max())
    df_max_cycle.reset_index(level=0, inplace=True)
    df_max_cycle.columns = ['id', 'last_cycle']

    # add time-to-failure ttf as a new column - regression label
    df_in = pd.merge(df_in, df_max_cycle, on='id')
    df_in['ttf'] = df_in['last_cycle'] - df_in['cycle']
    df_in.drop(['last_cycle'], axis=1, inplace='True')

    # create binary classification label
    df_in['label_bnc'] = df_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    # create multi-class classification label
    df_in['label_mcc'] = df_in['ttf'].apply(lambda x: 2 if x <= period / 2 else 1 if x <= period else 0)

    return df_in


def prepare_test_data(df_test_in, df_truth_in, period):
    """Add regression and classification labels to the test data.

        Regression label: ttf (time-to-failure) = extract the last cycle for each enginge and then merge the record with the truth data
        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)
        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

      Args:
          df_in (dataframe): The input training data
          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

      Returns:
          dataframe: The input dataframe with regression and classification labels added



    """

    df_tst_last_cycle = pd.DataFrame(df_test_in.groupby('id')['cycle'].max())

    df_tst_last_cycle.reset_index(level=0, inplace=True)
    df_tst_last_cycle.columns = ['id', 'last_cycle']

    df_test_in = pd.merge(df_test_in, df_tst_last_cycle, on='id')

    df_test_in = df_test_in[df_test_in['cycle'] == df_test_in['last_cycle']]

    df_test_in.drop(['last_cycle'], axis=1, inplace='True')

    df_test_in.reset_index(drop=True, inplace=True)

    df_test_in = pd.concat([df_test_in, df_truth], axis=1)

    # create binary classification label
    df_test_in['label_bnc'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    # create multi-class classification label
    df_test_in['label_mcc'] = df_test_in['ttf'].apply(lambda x: 2 if x <= period / 2 else 1 if x <= period else 0)

    return df_test_in

##################################
# Main Execution
##################################

def main():
    # Initialize DataLoader with Config
    data_loader = DataLoader(
        dataset_path=Config.DATASET_PATH,
        sequence_length=Config.SEQUENCE_LENGTH,
        w1=Config.W1,
        w0=Config.W0
    )
    data_loader.read_data()
    # OUTPUT_TYPE is now set in Config

    # Get training data
    train_df = data_loader.get_train_data()
    sequence_cols = data_loader.get_sequence_cols()
    nb_features = data_loader.get_nb_features()
    sequence_length = data_loader.get_sequence_length()
    output_type = data_loader.output_type

    # Generate sequences and labels for training
    seq_gen = SequenceGenerator(
        train_df, sequence_length, sequence_cols, data_loader.output_column
    )
    seq_array, label_array = seq_gen.generate_sequences()

    # Initialize and build model with parameters from Config
    pm_model = PredictiveMaintenanceModel(
        config=Config,
        nb_features=nb_features,
        output_type=output_type
    )
    pm_model.build_model()

    # Train model using parameters from Config
    history = pm_model.train_model(
        seq_array, label_array,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    # Plot training history
    plot_history(history, 'accuracy', Config.OUTPUT_PATH)
    plot_history(history, 'loss', Config.OUTPUT_PATH)

    # Evaluate on training data
    scores = pm_model.evaluate_model(seq_array, label_array, batch_size=Config.BATCH_SIZE)
    print(f'Training Accuracy: {scores[1]*100:.2f}%')

    # Predictions on training data
    y_pred_train = pm_model.predict(seq_array)
    y_true_train = label_array

    # Metrics for training data
    evaluate_performance(y_true_train, y_pred_train, output_type, 'Training')

    # Get test data
    test_df = data_loader.get_test_data()

    # Generate sequences and labels for test data
    seq_array_test, label_array_test = generate_test_sequences(
        test_df, sequence_length, sequence_cols, data_loader.output_column
    )

    # Load the best model
    pm_model.load_model()

    # Evaluate on test data
    scores_test = pm_model.evaluate_model(seq_array_test, label_array_test, batch_size=Config.BATCH_SIZE)
    print(f'Test Accuracy: {scores_test[1]*100:.2f}%')

    # Predictions on test data
    y_pred_test = pm_model.predict(seq_array_test)
    y_true_test = label_array_test

    # Metrics for test data
    evaluate_performance(y_true_test, y_pred_test, output_type, 'Test')

    # Plot predictions vs actual
    plot_predictions(y_true_test, y_pred_test, Config.OUTPUT_PATH)

if __name__ == "__main__":
    main()
