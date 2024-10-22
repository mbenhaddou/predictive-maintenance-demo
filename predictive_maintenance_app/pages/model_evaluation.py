# pages/model_evaluation.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from model.model_registry import MODEL_REGISTRY  # Ensure correct import
from model.predictive_maintenance_model import PredictiveMaintenanceModel
from model.sequence_generator import SequenceGenerator  # Adjust import as necessary
from model.data_loader import DataLoader  # Ensure DataLoader is correctly imported
from utils.helpers import instantiate_config  # Import helper functions if necessary
from utils.config import Config  # Base Config class
from model.lstm_model import LSTMConfig
from model.cnn_model import CNNConfig
import json
import os

def load_configuration(config_file_path: str) -> Config:
    """
    Load the configuration from a JSON file and instantiate the appropriate Config subclass.

    Args:
        config_file_path (str): Path to the configuration JSON file.

    Returns:
        Union[LSTMConfig, CNNConfig]: An instance of the appropriate Config subclass.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the config_class is unsupported or missing.
    """
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' does not exist.")

    with open(config_file_path, 'r') as f:
        config_data = json.load(f)

    config_class_name = config_data.get("config_class")
    model_name = config_data.get("MODEL_NAME")

    if not config_class_name:
        raise ValueError("The configuration file is missing the 'config_class' field.")

    # Instantiate the appropriate Config subclass
    if config_class_name == "LSTMConfig":
        config = LSTMConfig()
    elif config_class_name == "CNNConfig":
        config = CNNConfig()
    else:
        raise ValueError(f"Unsupported config_class '{config_class_name}' in configuration file.")

    # Remove keys that are not part of the Config subclass to avoid unexpected attributes
    ignored_keys = ["config_class", "MODEL_NAME"]
    for key, value in config_data.items():
        if key in ignored_keys:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            st.warning(f"Ignoring unknown configuration parameter: {key}")

    # Assign the MODEL_NAME to the config if needed
    config.MODEL_NAME = model_name

    return config

def display():
    """Display the Model Evaluation page."""
    st.header("Model Evaluation")
    st.write("Evaluating the model on test data.")

    # ---------------------------
    # 1. Load Configuration
    # ---------------------------
    st.subheader("1. Load Configuration")
    config_file_path = "configurations/config.json"  # Define the path to your configuration file

    try:
        config = load_configuration(config_file_path)
        st.success(f"Configuration loaded successfully from `{config_file_path}`!")
    except FileNotFoundError as fnf_error:
        st.error(str(fnf_error))
        st.stop()
    except ValueError as ve:
        st.error(str(ve))
        st.stop()
    except json.JSONDecodeError:
        st.error("The configuration file contains invalid JSON.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the configuration: {e}")
        st.stop()

    # Optionally, display the loaded configuration
    with st.expander("View Loaded Configuration"):
        config_dict = config.to_dict()
        config_dict["MODEL_NAME"] = config.MODEL_NAME  # Ensure MODEL_NAME is included
        st.json(config_dict)

    # ---------------------------
    # 2. Initialize DataLoader and Generate Sequences
    # ---------------------------
    st.subheader("2. Prepare Evaluation Data")
    try:
        data_loader = DataLoader(config)
        data_loader.read_data()
        nb_features = data_loader.get_nb_features()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    try:
        sequences, labels = data_loader.generate_test_sequences(config.SEQUENCE_LENGTH, config.OUTPUT_COLUMN)
        st.success(f"Generated {len(sequences)} sequences for evaluation.")
    except Exception as e:
        st.error(f"Error in generating sequences: {e}")
        st.stop()

    # ---------------------------
    # 3. Initialize and Load the Model
    # ---------------------------
    st.subheader("3. Load Trained Model")
    try:
        # Retrieve the selected model name from the config
        selected_model_name = getattr(config, 'MODEL_NAME', None)

        if not selected_model_name:
            st.error("Model name not found in the configuration. Please select a model in the configuration page.")
            st.stop()

        # Retrieve model information from the registry
        model_info = MODEL_REGISTRY.get(selected_model_name)
        if not model_info:
            st.error(f"Model '{selected_model_name}' not found in the MODEL_REGISTRY.")
            st.stop()

        model_class = model_info.get("model_class")
        if not model_class:
            st.error(f"Model class for '{selected_model_name}' not defined.")
            st.stop()

        # Instantiate the specific model class
        model_instance = model_class(
            config=config,
            nb_features=nb_features
        )

        # Wrap the model instance with PredictiveMaintenanceModel
        pm_model = PredictiveMaintenanceModel(
            model=model_instance,
            nb_features=nb_features,
            config=config
        )

        # Load the trained model
        pm_model.load_full_model()
        st.success("Trained model loaded successfully.")

        # Optionally, display model summary
        with st.expander("View Model Architecture"):
            model_summary = []
            pm_model.model.model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_str = "\n".join(model_summary)
            st.text(model_summary_str)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # ---------------------------
    # 4. Evaluate the Model on Test Data
    # ---------------------------
    st.subheader("4. Model Evaluation on Test Data")
    try:
        scores_test = pm_model.model.evaluate(
            np.array(sequences).astype(np.float32),
            np.array(labels).astype(np.float32),
            batch_size=config.BATCH_SIZE
        )
        st.write(f"**Test Loss:** {scores_test['loss']:.4f}")
        if len(scores_test) > 1:
            for metric_name, score in scores_test.items():
                if metric_name != 'loss':
                    st.write(f"**Test {metric_name}:** {score:.4f}")
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        st.stop()

    # ---------------------------
    # 5. Make Predictions on Test Data
    # ---------------------------
    st.subheader("5. Generate Predictions")
    try:
        if config.MODEL_TYPE == 'binary':
            y_pred_probs = pm_model.model.predict(sequences).flatten()
            threshold = getattr(config, 'BINARY_THRESHOLD', 0.5)
            y_pred = (y_pred_probs > threshold).astype(int)
        elif config.MODEL_TYPE == 'multiclass':
            y_pred_probs = pm_model.model.predict(sequences)
            y_pred = y_pred_probs.argmax(axis=1)
        elif config.MODEL_TYPE == 'regression':
            y_pred = pm_model.model.predict(sequences).flatten()
        else:
            st.error(f"Unsupported MODEL_TYPE '{config.MODEL_TYPE}'.")
            st.stop()
    except Exception as e:
        st.error(f"Error during predictions: {e}")
        st.stop()

    y_true = np.array(labels)

    # ---------------------------
    # 6. Display Performance Metrics
    # ---------------------------
    st.subheader("6. Performance Metrics")

    if config.MODEL_TYPE == 'binary':
        # Binary classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_probs)

        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")
        st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc_value = auc(fpr, tpr)
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash'))
        )
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title=f'ROC Curve (AUC = {roc_auc_value:.4f})'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    elif config.MODEL_TYPE == 'multiclass':
        # Multiclass classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision (Macro):** {precision:.4f}")
        st.write(f"**Recall (Macro):** {recall:.4f}")
        st.write(f"**F1-score (Macro):** {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

        # Plot ROC Curves for Multiclass
        st.subheader("ROC Curves for Each Class")

        # Binarize the output
        classes = np.unique(y_true)
        y_true_binarized = label_binarize(y_true, classes=classes)
        n_classes = y_true_binarized.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves for all classes
        fig_roc_multi = go.Figure()
        for i in range(n_classes):
            fig_roc_multi.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                mode='lines',
                name=f'Class {classes[i]} (AUC = {roc_auc[i]:.4f})'
            ))
        fig_roc_multi.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash')
        ))
        fig_roc_multi.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title='ROC Curves for Multiclass Classification'
        )
        st.plotly_chart(fig_roc_multi, use_container_width=True)

    elif config.MODEL_TYPE == 'regression':
        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
        st.write(f"**R² Score:** {r2:.4f}")

        # Plot Actual vs Predicted (Scatter Plot)
        st.subheader("Actual vs Predicted Values (Scatter Plot)")
        fig_reg_scatter = px.scatter(
            x=y_true,
            y=y_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title='Actual vs Predicted Values'
        )
        fig_reg_scatter.add_trace(go.Scatter(x=y_true, y=y_true, mode='lines', name='Ideal'))
        st.plotly_chart(fig_reg_scatter, use_container_width=True)

        # Plot Actual vs Predicted (Time Series)
        st.subheader("Actual vs Predicted Values (Time Series)")
        # Assuming that each sequence corresponds to a specific time point.
        # If you have actual timestamps, replace the x-axis accordingly.
        time_points = np.arange(len(y_true))  # Replace with actual timestamps if available

        fig_reg_time = go.Figure()
        fig_reg_time.add_trace(go.Scatter(
            x=time_points,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        fig_reg_time.add_trace(go.Scatter(
            x=time_points,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        fig_reg_time.update_layout(
            xaxis_title='Time',
            yaxis_title='Value',
            title='Actual vs Predicted Values Over Time'
        )
        st.plotly_chart(fig_reg_time, use_container_width=True)

    else:
        st.error(f"Unsupported MODEL_TYPE '{config.MODEL_TYPE}'.")

    # ---------------------------
    # 7. Optional: Save Evaluation Results
    # ---------------------------
    st.subheader("7. Save Evaluation Results")
    if st.button("Save Evaluation Metrics", key='save_evaluation_metrics_button'):
        try:
            evaluation_metrics = {}
            if config.MODEL_TYPE == 'binary':
                evaluation_metrics = {
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                    "ROC AUC Score": roc_auc
                }
            elif config.MODEL_TYPE == 'multiclass':
                evaluation_metrics = {
                    "Accuracy": accuracy,
                    "Precision (Macro)": precision,
                    "Recall (Macro)": recall,
                    "F1-score (Macro)": f1
                }
            elif config.MODEL_TYPE == 'regression':
                evaluation_metrics = {
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE": mape,
                    "R2 Score": r2
                }

            # Define the save path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_save_path = os.path.join("evaluation_results", f"{config.MODEL_NAME}_evaluation_{timestamp}.json")
            os.makedirs("evaluation_results", exist_ok=True)

            with open(metrics_save_path, 'w') as f:
                json.dump(evaluation_metrics, f, indent=4)

            st.success(f"✅ Evaluation metrics saved successfully at `{metrics_save_path}`!")
        except Exception as e:
            st.error(f"❌ Failed to save evaluation metrics: {e}")
