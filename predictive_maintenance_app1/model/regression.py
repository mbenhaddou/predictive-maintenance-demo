import os

from predictive_maintenance_app.utils.config import Config

seed = 42

import pandas as pd
import numpy as np
import random
import seaborn as sns; sns.set()
from predictive_maintenance_app.model.preprocessing import add_remaining_useful_life, gen_data_wrapper, gen_label_wrapper, prep_data, gen_test_data, evaluate
from predictive_maintenance_app.utils.config import Config
from predictive_maintenance_app.model.sequence_generator import SequenceGenerator
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, TimeDistributed

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_data(_config):
    """Load data using DataLoader."""
    from predictive_maintenance_app.model.data_loader import DataLoader  # Import here to avoid circular imports
    data_loader = DataLoader(_config)
    data_loader.read_data()
    data_loader.output_column = _config.OUTPUT_COLUMN
    return data_loader

config=Config()

config.OUTPUT_COLUMN = 'RUL'

data_loader = load_data(config)

# input_shape = (sequence_length, train_array.shape[2])
def create_model(input_shape, nodes_per_layer, dropout, activation, weights_file):
    model = Sequential()
    model.add(Masking(mask_value=-99., input_shape=input_shape))
    if len(nodes_per_layer) <= 1:
        model.add(LSTM(nodes_per_layer[0], activation=activation))
        model.add(Dropout(dropout))
    else:
        model.add(LSTM(nodes_per_layer[0], activation=activation, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes_per_layer[1], activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.save_weights(weights_file)
    model.summary()
    return model



remaining_sensors = ['s2', 's3', 's4', 's7', 's8', 's9',
       's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
drop_sensors = [element for element in data_loader.train_df.columns if element not in remaining_sensors+['id', 'cycle', 'voltage_input', 'current_limit', 'speed_control',  'RUL',
       'label_binary', 'label_multiclass']]

alpha = 0.1
sequence_length = 50
nodes_per_layer = [256]
dropout = 0.1
activation = 'sigmoid'
weights_file = 'fd001_model.weights.h5'
epochs = 15
batch_size = 128

# prep data
data_loader.prepare_data(drop_sensors, remaining_sensors, alpha)

#train_array = gen_data_wrapper(data_loader.train_df, sequence_length, remaining_sensors)
#label_array = gen_label_wrapper(data_loader.train_df, sequence_length, ['RUL'])

sequence_cols = data_loader.get_sequence_cols()
seq_gen = SequenceGenerator(
                df=data_loader.train_df,
                sequence_length=config.SEQUENCE_LENGTH,
                sequence_cols=sequence_cols,
                output_column=config.OUTPUT_COLUMN
            )
train_array, label_array = seq_gen.generate_sequences(padding_strategy='zero')


test_array, y_test= data_loader.generate_test_sequences(config.SEQUENCE_LENGTH, config.OUTPUT_COLUMN)

input_shape= (train_array.shape[1], train_array.shape[2])
final_model = create_model(input_shape, nodes_per_layer, dropout, activation, weights_file)

final_model.compile(loss='mean_squared_error', optimizer='adam')
final_model.load_weights(weights_file)

final_model.fit(np.array(train_array).astype(np.float32), np.array(label_array).astype(np.float32),
                epochs=epochs,
                batch_size=batch_size)

# predict and evaluate
y_hat_train = final_model.predict(np.array(train_array).astype(np.float32))
evaluate(np.array(label_array).astype(np.float32), y_hat_train, 'train')

y_hat_test = final_model.predict(np.array(test_array).astype(np.float32))
evaluate(y_test, y_hat_test)
