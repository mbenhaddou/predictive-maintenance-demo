import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

sns.set()


class PredictiveMaintenanceModel:
    def __init__(self, seed=42):
        # Set seeds for reproducibility
        self.seed = seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Initialize variables
        self.dir_path = '../../Dataset/'
        self.train_file = 'PM_train.txt'
        self.test_file = 'PM_test.txt'
        self.truth_file = 'PM_truth.txt'
        self.index_names = ['unit_nr', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = ['s_{}'.format(i + 1) for i in range(21)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names

        # Data placeholders
        self.train = None
        self.test = None
        self.y_test = None
        self.remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                                  's_11', 's_12', 's_13', 's_14', 's_15',
                                  's_17', 's_20', 's_21']
        self.drop_sensors = [sensor for sensor in self.sensor_names if sensor not in self.remaining_sensors]
        self.sequence_length = 30  # Default sequence length
        self.alpha = 0.1  # Default smoothing factor
        self.model = None
        self.history = None

    def load_data(self):
        # Load training data
        self.train = pd.read_csv(os.path.join(self.dir_path, self.train_file), sep='\s+', header=None,
                                 names=self.col_names)
        self.train = self.train.sort_values(['unit_nr', 'time_cycles'])

        # Load test data
        self.test = pd.read_csv(os.path.join(self.dir_path, self.test_file), sep='\s+', header=None,
                                names=self.col_names)
        self.test = self.test.sort_values(['unit_nr', 'time_cycles'])

        # Load RUL data for test set
        self.y_test = pd.read_csv(os.path.join(self.dir_path, self.truth_file), sep='\s+', header=None, names=['RUL'])
        self.y_test['unit_nr'] = self.y_test.index + 1 + 22510000

        print("Data loaded successfully.")

    def add_remaining_useful_life(self, df):
        # Calculate RUL
        df_max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        df_max_cycle.columns = ['unit_nr', 'max_cycle']
        df = df.merge(df_max_cycle, on='unit_nr', how='left')
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        df.drop('max_cycle', axis=1, inplace=True)
        return df

    def preprocess_data(self):
        # Add RUL to training data
        self.train = self.add_remaining_useful_life(self.train)
        self.train['RUL'].clip(upper=125, inplace=True)

        # Prepare test data RUL
        max_cycles = self.test.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_nr', 'max_cycle']
        self.y_test = self.y_test.merge(max_cycles, on='unit_nr', how='left')
        self.y_test['RUL'] = self.y_test['RUL'] + self.y_test['max_cycle']
        self.y_test.drop('max_cycle', axis=1, inplace=True)
        self.y_test.set_index('unit_nr', inplace=True)

        print("Data preprocessed successfully.")

    def add_operating_condition(self, df):
        df = df.copy()
        df['setting_1'] = df['setting_1'].round()
        df['setting_2'] = df['setting_2'].round(2)
        df['op_cond'] = df['setting_1'].astype(str) + '_' + df['setting_2'].astype(str) + '_' + df['setting_3'].astype(
            str)
        return df

    def condition_scaler(self, df_train, df_test):
        scaler = StandardScaler()
        for condition in df_train['op_cond'].unique():
            condition_mask_train = df_train['op_cond'] == condition
            condition_mask_test = df_test['op_cond'] == condition
            scaler.fit(df_train.loc[condition_mask_train, self.remaining_sensors])
            df_train.loc[condition_mask_train, self.remaining_sensors] = scaler.transform(
                df_train.loc[condition_mask_train, self.remaining_sensors])
            df_test.loc[condition_mask_test, self.remaining_sensors] = scaler.transform(
                df_test.loc[condition_mask_test, self.remaining_sensors])
        return df_train, df_test

    def exponential_smoothing(self, df, n_samples=0, alpha=0.4):
        df = df.copy()
        # Create an empty DataFrame to store the smoothed values
        smoothed_values = pd.DataFrame(index=df.index, columns=self.remaining_sensors)
        # Group by 'unit_nr' and apply exponential smoothing to each group
        for unit_nr, group in df.groupby('unit_nr'):
            smoothed_group = group[self.remaining_sensors].ewm(alpha=alpha).mean()
            smoothed_values.loc[group.index] = smoothed_group.values
        # Assign the smoothed values back to the DataFrame
        df[self.remaining_sensors] = smoothed_values
        if n_samples > 0:
            mask = df.groupby('unit_nr')['unit_nr'].transform(lambda x: np.arange(len(x)) >= n_samples)
            df = df[mask]
        return df

    def generate_sequences(self, df, sequence_length):
        data_gen = []
        label_gen = []
        unit_nrs = df['unit_nr'].unique()
        for unit_nr in unit_nrs:
            unit_data = df[df['unit_nr'] == unit_nr]
            data = unit_data[self.remaining_sensors].values.astype(np.float32)  # Ensure data type
            labels = unit_data['RUL'].values.astype(np.float32)  # Ensure data type
            num_elements = data.shape[0]
            for start, stop in zip(range(0, num_elements - sequence_length + 1),
                                   range(sequence_length, num_elements + 1)):
                data_gen.append(data[start:stop])
                label_gen.append(labels[stop - 1])
        data_array = np.array(data_gen, dtype=np.float32)
        label_array = np.array(label_gen, dtype=np.float32)
        return data_array, label_array

    def generate_test_sequences(self, df, sequence_length):
        data_gen = []
        unit_nrs = df['unit_nr'].unique()
        for unit_nr in unit_nrs:
            unit_data = df[df['unit_nr'] == unit_nr]
            data = unit_data[self.remaining_sensors].values.astype(np.float32)  # Ensure data type
            if data.shape[0] < sequence_length:
                # Pad sequences shorter than sequence_length
                padding = np.full((sequence_length - data.shape[0], len(self.remaining_sensors)), -99.,
                                  dtype=np.float32)
                data_padded = np.vstack((padding, data))
            else:
                data_padded = data[-sequence_length:]
            data_gen.append(data_padded)
        data_array = np.array(data_gen, dtype=np.float32)
        return data_array

    def build_model(self, input_shape, nodes_per_layer=[256], dropout=0.1, activation='sigmoid'):
        model = Sequential()
        model.add(Masking(mask_value=-99., input_shape=input_shape))
        for idx, nodes in enumerate(nodes_per_layer):
            return_sequences = idx < len(nodes_per_layer) - 1
            model.add(LSTM(nodes, activation=activation, return_sequences=return_sequences))
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        print("Model built successfully.")

    def train_model(self, X_train, y_train, epochs=15, batch_size=128):
        # Check data types
        print(f'X_train dtype: {X_train.dtype}')
        print(f'y_train dtype: {y_train.dtype}')

        # Check for NaN or None values
        print(f'X_train contains NaN: {np.isnan(X_train).any()}')
        print(f'y_train contains NaN: {np.isnan(y_train).any()}')
        print(f'X_train contains None: {np.any([x is None for x in X_train.flatten()])}')
        print(f'y_train contains None: {np.any([x is None for x in y_train.flatten()])}')

        early_stopping = EarlyStopping(monitor='loss', patience=5)
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                      callbacks=[early_stopping])
        print("Model trained successfully.")

    def evaluate_model(self, X, y_true, dataset='Train'):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_pred)
        print(f'{dataset} set RMSE: {rmse:.4f}, R2: {variance:.4f}')
        return y_pred

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def run(self):
        # Load data
        self.load_data()

        # Preprocess data
        self.preprocess_data()

        # Prepare training data
        X_train = self.add_operating_condition(self.train.drop(self.drop_sensors, axis=1))
        X_test = self.add_operating_condition(self.test.drop(self.drop_sensors, axis=1))
        X_train, X_test = self.condition_scaler(X_train, X_test)
        X_train = self.exponential_smoothing(X_train, alpha=self.alpha)
        X_test = self.exponential_smoothing(X_test, alpha=self.alpha)

        # Generate sequences
        train_array, label_array = self.generate_sequences(X_train, self.sequence_length)
        test_array = self.generate_test_sequences(X_test, self.sequence_length)
        y_test_array = self.y_test['RUL'].values

        # Build model
        input_shape = (self.sequence_length, len(self.remaining_sensors))
        self.build_model(input_shape, nodes_per_layer=[256], dropout=0.1, activation='sigmoid')

        # Train model
        self.train_model(train_array, label_array, epochs=15, batch_size=128)

        # Plot loss
        self.plot_loss()

        # Evaluate model
        print("\nEvaluating on Training Data:")
        self.evaluate_model(train_array, label_array, dataset='Train')

        print("\nEvaluating on Test Data:")
        self.evaluate_model(test_array, y_test_array, dataset='Test')


# Instantiate and run the model
if __name__ == "__main__":
    pm_model = PredictiveMaintenanceModel()
    pm_model.run()
