# utils/helpers.py

import json
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.metrics import (roc_curve, auc)
from sklearn.preprocessing import label_binarize

from utils.config import Config


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

def save_config(config, config_file='config.json'):
    """Save the current configuration to a file."""
    config_dict = config.to_dict()
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    st.success("Configuration saved successfully!")

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
