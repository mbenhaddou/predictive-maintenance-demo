# utils/helpers.py

import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, \
    recall_score, precision_score, f1_score
from sklearn.metrics import (roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List
from model.model_registry import MODEL_REGISTRY
from model.lstm_model import LSTMConfig
from model.cnn_model import CNNConfig
from utils.config import Config
import importlib
import streamlit as st

LABEL_TO_OUTPUT_TYPE = {
    "label_binary": "classification",
    "label_multiclass": "classification",
    "RUL": "regression"
}


@st.cache_data()
def load_csv(filepath):
    """Load a CSV file."""
    return pd.read_csv(filepath)

def assign_motors_to_engines(train_df, test_df, motors_df, seed=41):
    """Assign a random motor to each engine."""
    motor_ids = motors_df['Manufacturer Part Number'].unique()
    engine_ids = pd.concat([train_df['id'], test_df['id']]).unique()
    random.seed(seed)
    engine_motor_mapping = {engine_id: random.choice(motor_ids) for engine_id in engine_ids}
    train_df['Motor_ID'] = train_df['id'].map(engine_motor_mapping)
    test_df['Motor_ID'] = test_df['id'].map(engine_motor_mapping)
    return train_df, test_df, engine_motor_mapping

def save_config(config: LSTMConfig, config_file: str = 'config.json'):
    """
    Save the current configuration to a JSON file.

    Args:
        config (LSTMConfig): The configuration instance to save.
        config_file (str): The filename to save the configuration to.
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        st.success(f"Configuration saved successfully to {config_file}!")
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")

def load_or_initialize_config(config_file='config.json'):
    """Load existing configuration or initialize with defaults."""
    config = Config()
    if os.path.exists(config_file):
        config.load_from_file(config_file)
    return config

def plot_sensor_readings(historical_data, cycles, selected_features, feature_descriptions):
    """Plot sensor readings over cycles."""
    fig = go.Figure()
    for feature in selected_features:
        feature_label = feature_descriptions.get(feature, feature)
        fig.add_trace(go.Scatter(
            x=cycles,
            y=historical_data[feature],
            mode='lines+markers',
            name=feature_label
        ))
    fig.update_layout(
        xaxis_title='Cycle',
        yaxis_title='Sensor Readings',
        showlegend=True,
        height=250,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    return fig

def plot_confusion_matrix(corr_matrix):
    """Plot confusion matrix using Seaborn."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_xlabel('Features')
    ax.set_ylabel('Features')
    plt.tight_layout()
    return fig

def plot_roc_curve_binary(fpr, tpr, roc_auc):
    """Plot ROC curve for binary classification."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title=f'ROC Curve (AUC = {roc_auc:.4f})'
    )
    return fig

def plot_multiclass_roc(y_true, y_pred_probs, classes):
    """Plot ROC curves for multiclass classification."""
    fig = go.Figure()
    for i, cls in enumerate(classes):
        y_true_binary = label_binarize(y_true, classes=classes)[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {cls} (AUC = {roc_auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='ROC Curves for Multiclass Classification'
    )
    return fig

def plot_regression_predictions(y_true, y_pred):
    """Plot actual vs predicted values for regression."""
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted Values'
    )
    fig.add_trace(go.Scatter(x=y_true, y=y_true, mode='lines', name='Ideal'))
    fig.update_layout(
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=400
    )
    return fig

def initialize_placeholders(engine_ids):
    """Initialize placeholders for each engine in multiple engines simulation."""
    placeholders = {}
    for engine_id in engine_ids:
        engine_container = st.container()
        with engine_container:
            cols = st.columns([0.5, 2, 0.8, 2])  # Adjusted column widths

            with cols[0]:
                status_placeholder = st.empty()
                # Default status
                status_indicator = '🟢'
                status_placeholder.write(f"{status_indicator} **Engine ID: {engine_id}**")

            sensor_placeholder = cols[1].empty()
            gauge_placeholder = cols[2].empty()
            prediction_placeholder = cols[3].empty()

            placeholders[engine_id] = {
                'status_placeholder': status_placeholder,
                'sensor_placeholder': sensor_placeholder,
                'gauge_placeholder': gauge_placeholder,
                'prediction_placeholder': prediction_placeholder,
                'container': engine_container,
                'cols': cols
            }
    return placeholders



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

    df_test_in = pd.concat([df_test_in, df_truth_in], axis=1)

    # create binary classification label
    df_test_in['label_bnc'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    # create multi-class classification label
    df_test_in['label_mcc'] = df_test_in['ttf'].apply(lambda x: 2 if x <= period / 2 else 1 if x <= period else 0)

    return df_test_in

def get_models_by_task_type(model_type: str) -> List[str]:
    """
    Retrieve a list of model names from MODEL_REGISTRY that match the given output type.

    Args:
        model_type (str): The output type to filter models by (e.g., 'binary', 'multiclass', 'regression').

    Returns:
        List[str]: A list of model names that match the output type.
    """
    matching_models = []
    for model_name, model_info in MODEL_REGISTRY.items():
        if model_info.get("config_overrides", {}).get("TASK_TYPE", "").lower() == model_type.lower():
            matching_models.append(model_name)
    return matching_models

def apply_model_overrides(config: Config, model_name: str):
    """
    Apply configuration overrides from the MODEL_REGISTRY to the given config instance.

    Args:
        config (Config or LSTMConfig or CNNConfig): The configuration instance to update.
        model_name (str): The name of the model whose overrides are to be applied.
    """
    model_info = MODEL_REGISTRY.get(model_name, {})
    config_overrides = model_info.get("config_overrides", {})
    config.update_from_dict(config_dict=config_overrides)

def instantiate_config(config_class_name: str):
    """
    Instantiate the appropriate configuration class based on the class name.

    Args:
        config_class_name (str): The name of the configuration class.

    Returns:
        An instance of the specified configuration class.

    Raises:
        ValueError: If the configuration class name is unsupported.
    """
    if config_class_name == "LSTMConfig":
        return LSTMConfig()
    elif config_class_name == "CNNConfig":
        return CNNConfig()
    else:
        raise ValueError(f"Unsupported configuration class: {config_class_name}")


def save_config(config: Config, config_file: str = 'config.json'):
    """
    Save the current configuration to a JSON file.

    Args:
        config (Config or LSTMConfig or CNNConfig): The configuration instance to save.
        config_file (str): The filename to save the configuration to.
    """
    try:
        config.save_to_file(config_file)
        st.success(f"Configuration saved successfully to {config_file}!")
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")