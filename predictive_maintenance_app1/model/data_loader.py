
# data_loader.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, label_binarize
from predictive_maintenance_app1.model.preprocessing import *
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataLoader:
    """
    Class for loading and preprocessing data.
    """
    def __init__(self, config):
        """
        Initializes the DataLoader with a Config instance.

        Args:
            config (Config): An instance of the Config class containing configuration parameters.
        """
        self.config = config  # Store the config instance
        self.dataset_path = self.config.DATASET_PATH
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.w1 = self.config.W1
        self.w0 = self.config.W0
        self.scaler = MinMaxScaler()
        self.train_df = None
        self.test_df = None
        self.truth_df = None
        self.labels = []
        self.nb_features = None
        self.nb_out = 1



    @property
    def output_column(self):
        return self.config.OUTPUT_COLUMN

    @property
    def output_type(self):
        return self.config.MODEL_TYPE

    @output_type.setter
    def output_type(self, value):
        self.config.MODEL_TYPE = value

    @output_column.setter
    def output_column(self, value):
        self.config.OUTPUT_COLUMN = value
    def detect_output_type(self, df):
        """
        Detects the output type based on the output column values.
        """
        if pd.api.types.is_numeric_dtype(df[self.output_column]):
            unique_values = df[self.output_column].nunique()
            if unique_values == 2:
                self.output_type = 'binary'
            elif 2 < unique_values < 10:  # Adjust threshold as needed
                self.output_type = 'multiclass'
            else:
                self.output_type = 'regression'
            self.config.MODEL_TYPE = self.output_type  # Update config instance
            logger.info(f"Detected output type: {self.output_type}")
        else:
            raise ValueError(f"Output column '{self.output_column}' has non-numeric values, which is unsupported.")

    def prepare_data(self, drop_sensors, remaining_sensors, alpha):
        self.train_df = add_operating_condition(self.train_df.drop(drop_sensors, axis=1))
        self.test_df = add_operating_condition(self.test_df.drop(drop_sensors, axis=1))

        self.train_df, X_test_interim = condition_scaler(self.train_df, self.test_df, remaining_sensors)

        self.train_df = exponential_smoothing(self.train_df, remaining_sensors, 0, alpha)
        self.test_df = exponential_smoothing(X_test_interim, remaining_sensors, 0, alpha)



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
            self.train_df['RUL'].clip(upper=125, inplace=True)
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
        df.columns = ['id', 'cycle', 'voltage_input', 'current_limit', 'speed_control'] + [f's{i}' for i in range(1, 22)]
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
        self._sequence_cols = ['voltage_input', 'current_limit', 'speed_control', 'cycle'] + sensor_cols
        self.nb_features = len(self.sequence_cols)
    @property
    def sequence_cols(self):
        return [col for col in self.train_df.columns.values.tolist() if col not in self.labels+[self.config.ID_COLUMN, self.config.TIMESTEP_COLUMN,'op_cond']]
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


        df['label_binary'] = np.where(df['RUL'] <= self.w1, 1, 0)
        self.labels.append('label_binary')
        df['label_multiclass'] = 0
        self.labels.append('label_multiclass')
        df.loc[df['RUL'] <= self.w1, 'label_multiclass'] = 1
        df.loc[df['RUL'] <= self.w0, 'label_multiclass'] = 2
        self.labels.append('RUL')
        return df

    def generate_test_sequences(self, sequence_length, output_column):
        """
        Generates sequences and labels for test data.
        """
        seq_array_test = []
        label_array_test = []
        for id in self.test_df['id'].unique():
            id_df = self.test_df[self.test_df['id'] == id]
            if len(id_df) >= sequence_length:
                seq = id_df[self.get_sequence_cols()].values[-sequence_length:]
                seq_array_test.append(seq)
                label = id_df[output_column].values[-1]
                label_array_test.append(label)
        return np.array(seq_array_test), np.array(label_array_test)

    def _normalize(self, df, scaler=None):
        """
        Normalizes the dataframe using MinMaxScaler.
        """
        cols_normalize = df.columns.difference \
            (['id', 'cycle', 'RUL', 'label_binary', 'label_multiclass', 'RUL'])
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
        #max_cycle = self.test_df.groupby('id')['cycle'].max().reset_index()
        #max_cycle.columns = ['id', 'max_cycle']
        # Assuming truth_df has 'RUL' corresponding to each 'id'
        # Adjust if 'truth_df' structure is different
        self.truth_df['id'] = self.truth_df.index + 1 + 22510000 # Adjust if IDs start from a different number
        self.test_df = self.test_df.merge(self.truth_df[['id', 'RUL']], on='id', how='left')
        #self.test_df['RUL'] = self.test_df['RUL']
        self.test_df = self._generate_labels(self.test_df)
        return self.test_df

    def get_train_data(self):
        """
        Returns the training dataframe.
        """
        return self.train_df[self.sequence_cols]

    def get_test_data(self):
        """
        Returns the test dataframe.
        """
        return self.test_df[self.sequence_cols]

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
