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

from model import (
    PredictiveMaintenanceModel,
    SequenceGenerator,
    DataLoader  # Ensure DataLoader is correctly imported
)

def display(config):
    """Display the Model Evaluation page."""
    st.header("Model Evaluation")
    st.write("Evaluating the model on test data.")

    # Initialize DataLoader with the current config
    try:
        data_loader = DataLoader(config)
        data_loader.read_data()
        nb_features = data_loader.get_nb_features()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Generate sequences and labels for test data

    try:
        sequences, labels = data_loader.generate_test_sequences(config.SEQUENCE_LENGTH, config.OUTPUT_COLUMN)
        st.success(f"Generated {len(sequences)} sequences for evaluation.")
    except Exception as e:
        st.error(f"Error in generating sequences: {e}")
        return

    # Initialize and load the predictive maintenance model
    try:
        pm_model = PredictiveMaintenanceModel(config, nb_features, config.MODEL_TYPE)
        pm_model.load_and_build_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Evaluate on test data
    try:
        scores_test = pm_model.model.evaluate(np.array(sequences), np.array(labels), batch_size=config.BATCH_SIZE, verbose=0)
        st.write(f"Test Loss: {scores_test[0]:.4f}")
        if len(scores_test) > 1:
            for metric_name, score in zip(pm_model.model.metrics_names[1:], scores_test[1:]):
                st.write(f"Test {metric_name}: {score:.4f}")
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        return

    # Predictions on test data
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
    except Exception as e:
        st.error(f"Error during predictions: {e}")
        return

    y_true = np.array(labels)

    st.subheader("Performance Metrics")

    if config.MODEL_TYPE == 'binary':
        # Binary classification metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_probs)

        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-score: {f1:.4f}")
        st.write(f"ROC AUC Score: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
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
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision (Macro): {precision:.4f}")
        st.write(f"Recall (Macro): {recall:.4f}")
        st.write(f"F1-score (Macro): {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
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

        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        st.write(f"R^2 Score: {r2:.4f}")

        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Values")
        fig_reg = px.scatter(
            x=y_true,
            y=y_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title='Actual vs Predicted Values'
        )
        fig_reg.add_trace(go.Scatter(x=y_true, y=y_true, mode='lines', name='Ideal'))
        st.plotly_chart(fig_reg, use_container_width=True)
