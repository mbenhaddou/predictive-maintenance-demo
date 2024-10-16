import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys, logging, random, time
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error, roc_auc_score)
from sklearn.preprocessing import label_binarize

sys.path.append('../')
st.set_page_config(layout='wide')

from src.lstm.predictive_maintenance import (
    DataLoader, SequenceGenerator, PredictiveMaintenanceModel, Config,
    plot_history, evaluate_performance, generate_test_sequences, plot_predictions
)
from src.lstm.callbacks import StreamlitCallback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Helper Functions
# ---------------------------

@st.cache_data()
def load_csv(filepath):
    """Load a CSV file."""
    return pd.read_csv(filepath)

def assign_motors_to_engines(train_df, test_df, motors_df):
    """Assign a random motor to each engine."""
    motor_ids = motors_df['Manufacturer Part Number'].unique()
    engine_ids = pd.concat([train_df['id'], test_df['id']]).unique()
    random.seed(41)
    engine_motor_mapping = {engine_id: random.choice(motor_ids) for engine_id in engine_ids}
    train_df['Motor_ID'] = train_df['id'].map(engine_motor_mapping)
    test_df['Motor_ID'] = test_df['id'].map(engine_motor_mapping)
    return train_df, test_df, engine_motor_mapping

def load_or_initialize_config(config_file='config.json'):
    """Load existing configuration or initialize with defaults."""
    config = Config()
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            config.update_from_dict(config_data)
    # Ensure essential paths are set
    config.DATASET_PATH = '../Dataset/'
    config.OUTPUT_PATH = '../Output/'
    return config

def save_config(config, config_file='config.json'):
    """Save the current configuration to a file."""
    config_dict = config.to_dict()
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    st.success("Configuration saved successfully!")

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

def plot_confusion_matrix(cm):
    """Plot confusion matrix using Seaborn."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
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
        fpr, tpr, _ = roc_curve(label_binarize(y_true, classes=classes)[:, i], y_pred_probs[:, i])
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

# ---------------------------
# Main Application Functions
# ---------------------------

def introduction():
    """Display the introduction page."""
    st.header("Introduction")
    st.write("""
        This application demonstrates the use of LSTM neural networks for predictive maintenance.
        By analyzing historical engine data, the model predicts potential failures before they occur.
    """)
    st.sidebar.info("Use the navigation menu to explore different functionalities of the app.")
    # Optionally, add an image if available
    # st.image("engine_image.jpg", use_column_width=True)


def data_exploration(config, train_df, test_df, motors_df, engine_motor_mapping, sensors_df):
    """Display the data exploration page."""
    st.header("Data Exploration")
    st.subheader("Select an Engine ID to Explore")
    engine_ids = train_df['id'].unique()
    selected_id = st.selectbox("Engine ID", engine_ids)

    # Filter data for the selected engine ID
    engine_data = train_df[train_df['id'] == selected_id]

    # Get the motor assigned to this engine
    motor_id = engine_motor_mapping[selected_id]
    motor_spec = motors_df[motors_df['Manufacturer Part Number'] == motor_id].iloc[0]

    st.write(f"Showing data for Engine ID: {selected_id}")

    # Display motor specifications and image together
    st.write("### Motor Information:")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("#### Specifications")
        st.dataframe(motor_spec.to_frame())

    with col2:
        st.write("#### Motor Image")
        motor_name = motor_spec['Brand']
        image_extensions = ['.png']
        image_found = False
        for ext in image_extensions:
            image_path = f'../Dataset/images/{motor_name}{ext}'
            if os.path.exists(image_path):
                st.image(image_path, caption=motor_name, use_column_width=True)
                image_found = True
                break
        if not image_found:
            st.write(f"No image found for motor '{motor_name}'.")

    st.write("### Engine Data Preview:")
    st.dataframe(engine_data.head())

    # Time Series Plot of Sensors
    st.subheader("Time Series Plot of Sensors for Selected Engine")
    sensor_options = [col for col in train_df.columns if col in sensors_df['Sensor Name'].values]
    sensor_display_options = [
        f"{sensor} - {sensors_df.loc[sensors_df['Sensor Name'] == sensor, 'Description'].values[0]}" for sensor in
        sensor_options]

    selected_sensors_display = st.multiselect(
        "Select Sensors to Plot",
        options=sensor_display_options,
        default=sensor_display_options[:3]
    )

    selected_sensors = [option.split(' - ')[0] for option in selected_sensors_display]

    if selected_sensors:
        fig = px.line(engine_data, x='cycle', y=selected_sensors,
                      labels={'value': 'Sensor Readings', 'variable': 'Sensor'},
                      title='Sensor Readings Over Time')
        for trace in fig.data:
            sensor_name = trace.name
            trace.name = sensors_df.loc[sensors_df['Sensor Name'] == sensor_name, 'Description'].values[0]
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one sensor to plot.")

    # Distribution of Sensor Readings
    st.subheader("Distribution of Sensor Readings")
    selected_sensor_hist_option = st.selectbox("Select Sensor for Histogram", sensor_display_options)
    selected_sensor_hist = selected_sensor_hist_option.split(' - ')[0]

    fig_hist = px.histogram(train_df, x=selected_sensor_hist, nbins=50,
                            title=f"Distribution of {sensors_df.loc[sensors_df['Sensor Name'] == selected_sensor_hist, 'Description'].values[0]}")
    fig_hist.update_layout(
        xaxis_title=f"{sensors_df.loc[sensors_df['Sensor Name'] == selected_sensor_hist, 'Description'].values[0]} ({selected_sensor_hist})",
        yaxis_title='Frequency'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Feature Selection Based on Correlation with Output
    st.header("Feature Selection Based on Output Correlation")

    # Output Type Selection
    output_options = [config.OUTPUT_COLUMN]
    selected_output = st.selectbox("Select Output Variable", output_options)

    # Calculate correlations with the selected output
    if config.OUTPUT_TYPE == "regression":
        # Use Pearson correlation for continuous output
        correlations = train_df[sensor_options + [selected_output]].corr()[selected_output].drop(selected_output)
    else:
        # Use point-biserial correlation for categorical outputs
        correlations = {}
        for feature in sensor_options:
            corr = train_df[feature].corr(train_df[selected_output], method='pearson')
            correlations[feature] = abs(corr)
        correlations = pd.Series(correlations)

    # Display correlations in a bar plot
    fig_corr_plot = px.bar(correlations, title=f"Correlation of Features with {selected_output}",
                           labels={'index': 'Feature', 'value': 'Correlation'})
    st.plotly_chart(fig_corr_plot)

    # Set a correlation threshold
    corr_threshold = st.slider("Correlation Threshold", min_value=0.0, max_value=1.0, value=0.3)
    selected_features_corr = correlations[correlations >= corr_threshold].index.tolist()
    st.write(f"Features selected based on correlation with {selected_output}: {selected_features_corr}")

    if selected_features_corr:
        # Filter data for selected features and add target for correlation matrix
        df_filtered_corr = train_df[selected_features_corr + [selected_output]]

        # Display correlation matrix including target label
        st.subheader(f"Correlation Matrix for Selected Features and {selected_output}")
        corr_matrix = df_filtered_corr.corr()

        fig_corr_matrix = plot_confusion_matrix(corr_matrix)
        st.pyplot(fig_corr_matrix)
    else:
        st.warning("No features meet the correlation threshold.")


def model_configuration(config):
    """Display the model configuration page."""
    st.header("Model Configuration")
    st.write("Adjust model parameters below:")

    # ---------------------------
    # 1. Output Type Selection
    # ---------------------------
    output_type_options = ["binary", "multiclass", "regression"]

    # Determine the index based on current OUTPUT_TYPE
    if config.OUTPUT_TYPE in output_type_options:
        output_type_index = output_type_options.index(config.OUTPUT_TYPE)
    else:
        output_type_index = 0  # Default to 'binary' if not found

    selected_output_type = st.selectbox(
        "Select Output Type",
        output_type_options,
        index=output_type_index,
        key='output_type_selectbox'
    )
    config.OUTPUT_TYPE = selected_output_type  # Update the config instance

    # ---------------------------
    # 2. Set Output Column
    # ---------------------------
    if selected_output_type == "binary":
        config.OUTPUT_COLUMN = "label_binary"
    elif selected_output_type == "multiclass":
        config.OUTPUT_COLUMN = "label_multiclass"
    elif selected_output_type == "regression":
        config.OUTPUT_COLUMN = "RUL"

    # ---------------------------
    # 3. Model Parameters
    # ---------------------------

    # Dataset Path
    config.DATASET_PATH = st.text_input(
        "Dataset Path",
        value=config.DATASET_PATH,
        key='dataset_path_input'
    )

    # Sequence Length
    config.SEQUENCE_LENGTH = st.number_input(
        "Sequence Length",
        min_value=10,
        max_value=100,
        value=config.SEQUENCE_LENGTH,
        step=10,
        key='sequence_length_input'
    )

    # Threshold W1 for Label Generation
    config.W1 = st.number_input(
        "Threshold W1 for Label Generation",
        min_value=1,
        max_value=100,
        value=config.W1,
        step=5,
        key='w1_input'
    )

    # Threshold W0 for Label Generation
    config.W0 = st.number_input(
        "Threshold W0 for Label Generation",
        min_value=1,
        max_value=100,
        value=config.W0,
        step=5,
        key='w0_input'
    )

    # LSTM Units for Layer 1 and Layer 2
    config.LSTM_UNITS = [
        st.number_input(
            "LSTM Units Layer 1",
            min_value=32,
            max_value=256,
            value=config.LSTM_UNITS[0],
            step=32,
            key='lstm_units_layer1'
        ),
        st.number_input(
            "LSTM Units Layer 2",
            min_value=32,
            max_value=256,
            value=config.LSTM_UNITS[1],
            step=32,
            key='lstm_units_layer2'
        )
    ]

    # Dropout Rates for Layer 1 and Layer 2
    config.DROPOUT_RATES = [
        st.slider(
            "Dropout Rate Layer 1",
            min_value=0.0,
            max_value=0.5,
            value=config.DROPOUT_RATES[0],
            step=0.1,
            key='dropout_rate_layer1'
        ),
        st.slider(
            "Dropout Rate Layer 2",
            min_value=0.0,
            max_value=0.5,
            value=config.DROPOUT_RATES[1],
            step=0.1,
            key='dropout_rate_layer2'
        )
    ]

    # L2 Regularization
    config.L2_REG = st.number_input(
        "L2 Regularization",
        min_value=0.0,
        max_value=0.01,
        value=config.L2_REG,
        step=0.001,
        format="%.3f",
        key='l2_reg_input'
    )

    # Optimizer Selection
    optimizer_options = ["adam", "sgd"]
    if config.OPTIMIZER in optimizer_options:
        optimizer_index = optimizer_options.index(config.OPTIMIZER)
    else:
        optimizer_index = 0  # Default to 'adam' if not found

    config.OPTIMIZER = st.selectbox(
        "Optimizer",
        optimizer_options,
        index=optimizer_index,
        key='optimizer_selectbox'
    )

    # Learning Rate
    config.LEARNING_RATE = st.number_input(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=config.LEARNING_RATE,
        step=0.0001,
        format="%.4f",
        key='learning_rate_input'
    )

    # Epochs
    config.EPOCHS = st.number_input(
        "Epochs",
        min_value=10,
        max_value=200,
        value=config.EPOCHS,
        step=10,
        key='epochs_input'
    )

    # Batch Size
    config.BATCH_SIZE = st.number_input(
        "Batch Size",
        min_value=64,
        max_value=512,
        value=config.BATCH_SIZE,
        step=64,
        key='batch_size_input'
    )

    # ---------------------------
    # 4. Validation
    # ---------------------------
    if config.W1 <= config.W0:
        st.error("W1 should be greater than W0 for proper label generation.")

    # ---------------------------
    # 5. Save Configuration Button
    # ---------------------------
    if st.button("Save Configuration", key='save_config_button'):
        save_config(config)

def model_training(config, train_df, sequence_cols, nb_features):
    """Display the model training page."""
    st.header("Model Training")
    st.write("Train your LSTM model with the configured parameters below.")

    # Generate sequences and labels for training
    seq_gen = SequenceGenerator(
        df=train_df,
        sequence_length=config.SEQUENCE_LENGTH,
        sequence_cols=sequence_cols,
        output_column=config.OUTPUT_COLUMN
    )
    seq_array, label_array = seq_gen.generate_sequences()

    st.subheader("Training Parameters")
    st.write("Review and adjust the training parameters as needed.")

    # Button to initiate training
    if st.button("Train Model", key='train_model_button'):
        try:
            # Initialize PredictiveMaintenanceModel
            pm_model = PredictiveMaintenanceModel(
                config=config,
                nb_features=nb_features,
                output_type=config.OUTPUT_TYPE
            )
            pm_model.build_model()

            # Placeholders for progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create the Streamlit callback with output_type
            st_callback = StreamlitCallback(
                epochs=config.EPOCHS,
                progress_bar=progress_bar,
                status_text=status_text,
                output_type=config.OUTPUT_TYPE
            )

            # Train model with the custom callback
            history = pm_model.train_model(
                seq_array=seq_array,
                label_array=label_array,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                custom_callback=st_callback  # Pass the custom callback
            )

            st.success("✅ Model training completed successfully!")

            # Plot training history based on output_type
            st.subheader("Training History")
            history_df = pd.DataFrame(history.history)

            # Display relevant plots based on the output type
            if config.OUTPUT_TYPE in ['binary', 'multiclass']:
                # Accuracy Plot
                if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
                    fig_acc = px.line(
                        history_df,
                        y=['accuracy', 'val_accuracy'],
                        labels={'index': 'Epoch', 'value': 'Accuracy', 'variable': 'Dataset'},
                        title='Model Accuracy Over Epochs'
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                else:
                    st.warning("⚠️ Accuracy metrics not available for the selected output type.")

            if config.OUTPUT_TYPE in ['binary', 'multiclass', 'regression']:
                # Loss Plot
                if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
                    fig_loss = px.line(
                        history_df,
                        y=['loss', 'val_loss'],
                        labels={'index': 'Epoch', 'value': 'Loss', 'variable': 'Dataset'},
                        title='Model Loss Over Epochs'
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.warning("⚠️ Loss metrics not available.")

            if config.OUTPUT_TYPE == 'regression':
                # Additional Regression Metrics Plots
                if 'mae' in history_df.columns and 'val_mae' in history_df.columns:
                    fig_mae = px.line(
                        history_df,
                        y=['mae', 'val_mae'],
                        labels={'index': 'Epoch', 'value': 'MAE', 'variable': 'Dataset'},
                        title='Model MAE Over Epochs'
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
                else:
                    st.warning("⚠️ MAE metrics not available.")

                if 'mse' in history_df.columns and 'val_mse' in history_df.columns:
                    fig_mse = px.line(
                        history_df,
                        y=['mse', 'val_mse'],
                        labels={'index': 'Epoch', 'value': 'MSE', 'variable': 'Dataset'},
                        title='Model MSE Over Epochs'
                    )
                    st.plotly_chart(fig_mse, use_container_width=True)
                else:
                    st.warning("⚠️ MSE metrics not available.")

        except Exception as e:
            st.error(f"❌ An error occurred during training: {e}")

def model_evaluation(config, test_df, sequence_cols, nb_features):
    """Display the model evaluation page."""
    st.header("Model Evaluation")
    st.write("Evaluating the model on test data.")

    # Initialize DataLoader with the current config
    data_loader = DataLoader(config)
    data_loader.read_data()

    # Generate sequences and labels for test data
    seq_array_test, label_array_test = generate_test_sequences(
        data_loader.get_test_data(), config.SEQUENCE_LENGTH, data_loader.get_sequence_cols(), config.OUTPUT_COLUMN
    )

    # Initialize and load the predictive maintenance model
    pm_model = PredictiveMaintenanceModel(config, data_loader.get_nb_features(), config.OUTPUT_TYPE)
    pm_model.load_and_build_model()

    # Evaluate on test data
    scores_test = pm_model.evaluate_model(seq_array_test, label_array_test, batch_size=config.BATCH_SIZE)

    if len(scores_test) > 1:
        st.write(f"Test Metric ({pm_model.metrics[0]}): {scores_test[pm_model.metrics[0]]:.4f}")
    else:
        st.write(f"Test Loss: {scores_test[0]:.4f}")

    # Predictions on test data
    if config.OUTPUT_TYPE == 'binary':
        y_pred_probs = pm_model.model.predict(seq_array_test).flatten()
        y_pred_test = (y_pred_probs > 0.5).astype(int)
    elif config.OUTPUT_TYPE == 'multiclass':
        y_pred_probs = pm_model.model.predict(seq_array_test)
        y_pred_test = np.argmax(y_pred_probs, axis=1)
    elif config.OUTPUT_TYPE == 'regression':
        y_pred_test = pm_model.model.predict(seq_array_test).flatten()

    y_true_test = label_array_test

    st.subheader("Performance Metrics")

    if config.OUTPUT_TYPE == 'binary':
        # Binary classification metrics
        precision = precision_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        roc_auc = roc_auc_score(y_true_test, y_pred_probs)

        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-score: {f1:.4f}")
        st.write(f"ROC AUC Score: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true_test, y_pred_test)
        st.write("Confusion Matrix:")
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm)

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_probs)
        roc_auc_value = auc(fpr, tpr)
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve_binary(fpr, tpr, roc_auc_value)
        st.plotly_chart(fig_roc, use_container_width=True)

    elif config.OUTPUT_TYPE == 'multiclass':
        # Multiclass classification metrics
        accuracy = accuracy_score(y_true_test, y_pred_test)
        precision = precision_score(y_true_test, y_pred_test, average='macro')
        recall = recall_score(y_true_test, y_pred_test, average='macro')
        f1 = f1_score(y_true_test, y_pred_test, average='macro')

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision (Macro): {precision:.4f}")
        st.write(f"Recall (Macro): {recall:.4f}")
        st.write(f"F1-score (Macro): {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true_test, y_pred_test)
        st.write("Confusion Matrix:")
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm)

        # Plot ROC Curves for Multiclass
        st.subheader("ROC Curves for Each Class")
        classes = config.class_labels if hasattr(config, 'class_labels') else ['Class_A', 'Class_B', 'Class_C']
        y_true_binarized = label_binarize(y_true_test, classes=classes)
        n_classes = y_true_binarized.shape[1]

        fig_roc_multi = plot_multiclass_roc(y_true_test, y_pred_probs, classes)
        st.plotly_chart(fig_roc_multi, use_container_width=True)

    elif config.OUTPUT_TYPE == 'regression':
        # Regression metrics
        mse = mean_squared_error(y_true_test, y_pred_test)
        mae = mean_absolute_error(y_true_test, y_pred_test)
        r2 = r2_score(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true_test, y_pred_test) * 100

        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        st.write(f"R^2 Score: {r2:.4f}")

        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Values")
        fig_reg = plot_regression_predictions(y_true_test, y_pred_test)
        st.plotly_chart(fig_reg, use_container_width=True)

def run_single_engine_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """Handle the single engine simulation."""
    st.subheader("Single Engine Simulation")
    st.write("Select an engine from the test set to simulate streaming data and generate predictions at each cycle.")

    # Select an engine ID from the test set
    engine_ids = test_df['id'].unique()
    selected_id = st.selectbox("Engine ID", engine_ids, key='single_engine_selectbox')

    # Get data for the selected engine ID
    engine_data = test_df[test_df['id'] == selected_id].reset_index(drop=True)

    # Get the motor assigned to this engine
    motor_id = engine_motor_mapping[selected_id]
    motor_spec = motors_df[motors_df['Manufacturer Part Number'] == motor_id].iloc[0]

    st.write(f"Selected Engine ID: {selected_id}")

    # Display motor specifications and image together
    st.write("### Motor Information:")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("#### Specifications")
        st.dataframe(motor_spec.to_frame())

    with col2:
        st.write("#### Motor Image")
        motor_name = motor_spec['Brand']
        image_extensions = ['.png']
        image_found = False
        for ext in image_extensions:
            image_path = f'../Dataset/images/{motor_name}{ext}'
            if os.path.exists(image_path):
                st.image(image_path, caption=motor_name, use_column_width=True)
                image_found = True
                break
        if not image_found:
            st.write(f"No image found for motor '{motor_name}'.")

    st.write("### Engine Data Preview:")
    st.dataframe(engine_data.head())

    # Validate Sequence Columns
    missing_cols = [col for col in sequence_cols if col not in engine_data.columns]
    if missing_cols:
        st.error(f"The following sequence columns are missing in the uploaded data: {missing_cols}")
        return

    # Generate Sequences with Padding using SequenceGenerator
    try:
        seq_gen = SequenceGenerator(
            df=engine_data,
            sequence_length=config.SEQUENCE_LENGTH,
            sequence_cols=sequence_cols,
            output_column=config.OUTPUT_COLUMN
        )
        sequences, labels = seq_gen.generate_sequences()
        total_sequences = len(sequences)
        st.success(f"Generated {total_sequences} sequences for prediction.")
    except Exception as e:
        st.error(f"Error in generating sequences: {e}")
        return

    # Prepare feature descriptions
    all_features = ['voltage_input', 'current_limit', 'speed_control'] + list(sensors_df['Sensor Name'])
    feature_descriptions = {
        **{feat: feat.replace('_', ' ').title() for feat in ['voltage_input', 'current_limit', 'speed_control']},
        **{row['Sensor Name']: f"{row['Sensor Name']} - {row['Description']}" for _, row in sensors_df.iterrows()}
    }

    # Allow users to select features to visualize
    selected_features = st.multiselect(
        "Select Features to Visualize",
        options=all_features,
        default=['voltage_input', 's1', 's2'],
        format_func=lambda x: feature_descriptions.get(x, x),
        help="Choose a subset of features to display in real-time.",
        key='single_engine_features'
    )

    if not selected_features:
        st.warning("Please select at least one feature to visualize.")
        return

    # Simulation Speed Slider with unique key
    speed = st.slider(
        "Select Simulation Speed (Seconds per Cycle)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Control how fast the simulation runs.",
        key='single_engine_speed_slider'
    )

    # Start/Stop Simulation
    start_button = st.button("Start Simulation", key='start_single_simulation')
    stop_button = st.button("Stop Simulation", key='stop_single_simulation')

    if start_button:
        st.session_state['run_single_simulation'] = True

    if stop_button:
        st.session_state['run_single_simulation'] = False

    if st.session_state['run_single_simulation']:
        # Initialize Placeholders for Real-Time Visualization
        input_placeholder = st.empty()
        output_placeholder = st.empty()

        # Initialize Lists to Store Predictions
        cycle_list = []
        y_pred_list = []
        historical_data = pd.DataFrame(columns=['Cycle'] + selected_features)

        # Load the model
        try:
            pm_model = PredictiveMaintenanceModel(config, nb_features, config.OUTPUT_TYPE)
            pm_model.load_and_build_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Define the classes list for multiclass classification
        if config.OUTPUT_TYPE == 'multiclass':
            classes = config.class_labels if hasattr(config, 'class_labels') else ['Class_A', 'Class_B', 'Class_C']
            num_classes = len(classes)
        elif config.OUTPUT_TYPE == 'binary':
            classes = [0, 1]
            num_classes = 2

        for i, (seq, label) in enumerate(zip(sequences, labels)):
            # Check if the simulation should continue
            if not st.session_state['run_single_simulation']:
                st.write("Simulation stopped.")
                break

            # Reshape the sequence to match model input
            seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)

            # Make Prediction
            try:
                if config.OUTPUT_TYPE == 'binary':
                    y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
                    y_pred_class = (y_pred_prob > 0.5).astype(int)
                elif config.OUTPUT_TYPE == 'multiclass':
                    y_pred_probs = pm_model.model.predict(seq_reshaped)[0]
                    y_pred_class = np.argmax(y_pred_probs)
                    y_pred_prob = y_pred_probs[y_pred_class]
                elif config.OUTPUT_TYPE == 'regression':
                    y_pred_value = pm_model.model.predict(seq_reshaped)[0][0]
            except Exception as e:
                st.error(f"Error during prediction at sequence {i + 1}: {e}")
                continue  # Skip to next iteration

            # Determine the Corresponding Cycle Number
            if i < len(engine_data):
                cycle_num = engine_data['cycle'].iloc[i]
            else:
                cycle_num = f"Cycle_{i + 1}"

            # Append to Lists
            cycle_list.append(cycle_num)
            if config.OUTPUT_TYPE == 'binary':
                y_pred_list.append(y_pred_prob)
            elif config.OUTPUT_TYPE == 'multiclass':
                y_pred_list.append(y_pred_probs)  # Store all class probabilities
            elif config.OUTPUT_TYPE == 'regression':
                y_pred_list.append(y_pred_value)

            # Update Historical Sensor Readings
            if cycle_num != f"Cycle_{i + 1}":
                current_features = seq[-1][[sequence_cols.index(feat) for feat in selected_features]]
                input_data = {feat: val for feat, val in zip(selected_features, current_features)}
                input_data['Cycle'] = cycle_num
                historical_data = historical_data._append(input_data, ignore_index=True)

            # Update Input Visualization with Historical Data (Sliding Window)
            with input_placeholder.container():
                st.write(f"**Cycle {cycle_num}**")
                if not historical_data.empty:
                    windowed_data = historical_data.tail(50)
                    fig_input = plot_sensor_readings(windowed_data[selected_features], windowed_data['Cycle'], selected_features, feature_descriptions)
                    st.plotly_chart(fig_input, use_container_width=True, key=f"input_{cycle_num}")
                else:
                    st.write("No historical data to display.")

            # Update Output Visualization
            with output_placeholder.container():
                if config.OUTPUT_TYPE == 'binary':
                    # Binary Classification: Use Gauge
                    if not np.isnan(y_pred_prob):
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=y_pred_prob * 100,  # Convert to percentage
                            title={'text': "Probability (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#00FF00" if y_pred_prob * 100 < 50 else "#FF0000"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#00FF00"},
                                    {'range': [50, 100], 'color': "#FF0000"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig_gauge.update_layout(width=600, height=400)
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_binary_{cycle_num}")
                    else:
                        st.write("Prediction Error")

                elif config.OUTPUT_TYPE == 'multiclass':
                    # Multiclass Classification: Use Bar Chart
                    y_pred_probs_full = y_pred_probs  # Already obtained above

                    # Validate that classes and y_pred_probs_full have the same length
                    if len(classes) != len(y_pred_probs_full):
                        st.error("Mismatch between number of classes and predicted probabilities.")
                        st.stop()

                    fig_bar = px.bar(
                        x=classes,
                        y=y_pred_probs_full * 100,  # Convert to percentage
                        labels={'x': 'Classes', 'y': 'Probability (%)'},
                        title=f"Class Probabilities at Cycle {cycle_num}",
                        text=y_pred_probs_full * 100
                    )
                    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig_bar.update_layout(yaxis=dict(range=[0, 100]), showlegend=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_multiclass_{cycle_num}")

                elif config.OUTPUT_TYPE == 'regression':
                    # Regression: Display predicted value
                    st.write(f"Predicted Value at Cycle {cycle_num}: {y_pred_value:.2f}")

            # Simulate real-time by adding a delay
            time.sleep(speed)

        # Create a DataFrame for All Predictions
        if config.OUTPUT_TYPE == 'binary':
            pred_df = pd.DataFrame({
                'Cycle': cycle_list,
                'Predicted_Probability': y_pred_list
            })
        elif config.OUTPUT_TYPE == 'multiclass':
            # Convert list of numpy arrays to separate columns
            pred_df = pd.DataFrame(cycle_list, columns=['Cycle'])
            for cls_idx, cls in enumerate(classes):
                pred_df[f'Probability_{cls}'] = [prob[cls_idx] * 100 for prob in y_pred_list]
        elif config.OUTPUT_TYPE == 'regression':
            pred_df = pd.DataFrame({
                'Cycle': cycle_list,
                'Predicted_Value': y_pred_list
            })

        # Handle NaN Predictions
        pred_df.dropna(inplace=True)

        # Display the Predictions DataFrame
        st.subheader("Prediction Results")
        st.dataframe(pred_df)

        # Plot the Predictions Over Time
        st.subheader(f"Predictions Over Time for Engine {selected_id}")
        if config.OUTPUT_TYPE == 'binary':
            fig_pred = px.line(
                pred_df,
                x='Cycle',
                y='Predicted_Probability',
                title=f"Predicted Probability Over Cycles for Engine {selected_id}"
            )
            fig_pred.add_hline(y=50, line_dash='dash', line_color='red',
                               annotation_text='Threshold (50%)')
            fig_pred.update_yaxes(title_text='Predicted Probability (%)')
            fig_pred.update_xaxes(title_text='Cycle')
            st.plotly_chart(fig_pred, use_container_width=True, key=f"pred_binary_{selected_id}")

        elif config.OUTPUT_TYPE == 'multiclass':
            # Plot probabilities for all classes
            prob_columns = [f'Probability_{cls}' for cls in classes]
            fig_pred = px.line(
                pred_df,
                x='Cycle',
                y=prob_columns,
                title=f"Predicted Class Probabilities Over Cycles for Engine {selected_id}"
            )
            fig_pred.update_yaxes(title_text='Predicted Probability (%)')
            fig_pred.update_xaxes(title_text='Cycle')
            st.plotly_chart(fig_pred, use_container_width=True, key=f"pred_multiclass_{selected_id}")

        elif config.OUTPUT_TYPE == 'regression':
            fig_pred = px.line(
                pred_df,
                x='Cycle',
                y='Predicted_Value',
                title=f"Predicted Values Over Cycles for Engine {selected_id}"
            )
            fig_pred.update_yaxes(title_text='Predicted Value')
            fig_pred.update_xaxes(title_text='Cycle')
            st.plotly_chart(fig_pred, use_container_width=True, key=f"pred_regression_{selected_id}")

        # Optionally, Allow Users to Download the Predictions
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

def run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """Handle the multiple engines simulation."""
    st.subheader("Multiple Engines Simulation")
    st.write("Simulate streaming data and predictions for multiple engines simultaneously.")

    # Allow user to select number of engines (default is 10)
    num_engines = st.number_input(
        "Select Number of Engines to Simulate",
        min_value=1,
        max_value=min(10, len(test_df['id'].unique())),
        value=10,
        step=1,
        key='multi_engine_num_engines'
    )

    # Option for random selection
    random_selection = st.checkbox("Randomly Select Engines", value=True, key='multi_engine_random_selection')

    if random_selection:
        # Ensure that the number of engines to simulate does not exceed available unique engines
        available_engines = list(test_df['id'].unique())
        if num_engines > len(available_engines):
            st.warning(f"Only {len(available_engines)} unique engines available. Adjusting the number of engines to simulate.")
            num_engines = len(available_engines)
        selected_ids = random.sample(available_engines, num_engines)
    else:
        # Allow user to select specific engines
        selected_ids = st.multiselect(
            "Select Engine IDs",
            options=test_df['id'].unique(),
            default=test_df['id'].unique()[:num_engines],
            key='multi_engine_selected_ids'
        )

    if not selected_ids:
        st.warning("Please select at least one engine to simulate.")
        return

    # Feature Selection Step
    all_features = ['voltage_input', 'current_limit', 'speed_control'] + list(sensors_df['Sensor Name'])

    feature_descriptions = {
        **{feat: feat.replace('_', ' ').title() for feat in ['voltage_input', 'current_limit', 'speed_control']},
        **{row['Sensor Name']: f"{row['Sensor Name']} - {row['Description']}" for _, row in sensors_df.iterrows()}
    }

    selected_features = st.multiselect(
        "Select Features to Visualize for All Engines",
        options=all_features,
        default=['voltage_input', 's1', 's2'],
        format_func=lambda x: feature_descriptions.get(x, x),
        help="Choose a subset of features to display in real-time across all engines.",
        key='multi_engine_features'
    )

    if not selected_features:
        st.warning("Please select at least one feature to visualize.")
        return

    # Simulation Speed Slider with unique key
    speed = st.slider(
        "Select Simulation Speed (Seconds per Cycle)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Control how fast the simulation runs.",
        key='multi_engine_speed_slider'
    )

    # Start/Stop Simulation
    start_button = st.button("Start Simulation", key='start_multiple_simulation')
    stop_button = st.button("Stop Simulation", key='stop_multiple_simulation')

    if start_button:
        st.session_state['run_multiple_simulation'] = True

    if stop_button:
        st.session_state['run_multiple_simulation'] = False

    if st.session_state['run_multiple_simulation']:
        # Load the model
        try:
            pm_model = PredictiveMaintenanceModel(config, nb_features, config.OUTPUT_TYPE)
            pm_model.load_and_build_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Prepare data for selected engines
        engines_data = {}
        sequences_dict = {}
        labels_dict = {}
        max_cycles = 0

        for engine_id in selected_ids:
            engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
            # Generate sequences
            try:
                seq_gen = SequenceGenerator(
                    df=engine_data,
                    sequence_length=config.SEQUENCE_LENGTH,
                    sequence_cols=sequence_cols,
                    output_column=config.OUTPUT_COLUMN  # Ensure output_column is provided
                )
                sequences, labels = seq_gen.generate_sequences()
            except Exception as e:
                st.error(f"Error generating sequences for engine {engine_id}: {e}")
                continue  # Skip this engine if sequence generation fails

            sequences_dict[engine_id] = sequences
            labels_dict[engine_id] = labels
            engines_data[engine_id] = engine_data
            max_cycles = max(max_cycles, len(sequences))

        # Define classes for multiclass if applicable
        if config.OUTPUT_TYPE == 'multiclass':
            classes = config.class_labels if hasattr(config, 'class_labels') else ['Class_A', 'Class_B', 'Class_C']
            num_classes = len(classes)

        # Initialize DataFrames to Store Predictions
        pred_dfs = {}
        for engine_id in selected_ids:
            if config.OUTPUT_TYPE in ['binary', 'multiclass']:
                if config.OUTPUT_TYPE == 'binary':
                    columns = ['Cycle', 'Predicted_Probability']
                else:
                    # For multiclass, add a column for each class probability
                    columns = ['Cycle'] + [f'Probability_{cls}' for cls in classes]
                pred_dfs[engine_id] = pd.DataFrame(columns=columns)
            elif config.OUTPUT_TYPE == 'regression':
                pred_dfs[engine_id] = pd.DataFrame(columns=['Cycle', 'Predicted_Value'])

        # Initialize placeholders for each engine
        engine_placeholders = initialize_placeholders(selected_ids)

        # Simulation Loop
        for cycle_idx in range(max_cycles):
            # Check if the simulation should continue
            if not st.session_state['run_multiple_simulation'] or st.session_state['active_tab'] != "Multiple Engines Simulation":
                st.write("Simulation stopped.")
                break

            # For each engine, update the displays
            for engine_id in selected_ids:
                sequences = sequences_dict.get(engine_id, [])
                engine_data = engines_data.get(engine_id, pd.DataFrame())
                placeholders = engine_placeholders.get(engine_id, {})
                if not placeholders:
                    continue  # Skip if placeholders are not initialized

                if cycle_idx < len(sequences):
                    seq = sequences[cycle_idx]
                    # Reshape sequence
                    seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)
                    # Make prediction
                    try:
                        if config.OUTPUT_TYPE == 'binary':
                            y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
                            y_pred_class = (y_pred_prob > config.BINARY_THRESHOLD).astype(int)
                        elif config.OUTPUT_TYPE == 'multiclass':
                            y_pred_probs = pm_model.model.predict(seq_reshaped)[0]
                            y_pred_class = np.argmax(y_pred_probs)
                            # y_pred_prob is the probability of the predicted class
                            y_pred_prob = y_pred_probs[y_pred_class]
                        elif config.OUTPUT_TYPE == 'regression':
                            y_pred_value = pm_model.model.predict(seq_reshaped)[0][0]
                    except Exception as e:
                        st.error(
                            f"Error during prediction for engine {engine_id} at cycle {cycle_idx + 1}: {e}")
                        continue  # Skip to next iteration

                    # Get cycle number
                    cycle_num = engine_data['cycle'].iloc[cycle_idx] if cycle_idx < len(engine_data) else f"Cycle_{cycle_idx+1}"

                    # Append prediction to DataFrame
                    if config.OUTPUT_TYPE == 'binary':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'Cycle': cycle_num,
                            'Predicted_Probability': y_pred_prob * 100  # Convert to percentage
                        }, ignore_index=True)
                    elif config.OUTPUT_TYPE == 'multiclass':
                        pred_entry = {'Cycle': cycle_num}
                        for cls_idx, cls in enumerate(classes):
                            pred_entry[f'Probability_{cls}'] = y_pred_probs[cls_idx] * 100  # Convert to percentage
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append(pred_entry, ignore_index=True)
                    elif config.OUTPUT_TYPE == 'regression':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'Cycle': cycle_num,
                            'Predicted_Value': y_pred_value
                        }, ignore_index=True)

                    # **Update Status Indicator Based on Predicted Class**
                    with placeholders['status_placeholder']:
                        if config.OUTPUT_TYPE == 'binary':
                            threshold = config.BINARY_THRESHOLD * 100  # Convert to percentage
                            if y_pred_prob * 100 < threshold:
                                status_indicator = config.STATUS_COLORS.get('ClassA', '🟢')  # Safe
                            else:
                                status_indicator = config.STATUS_COLORS.get('ClassC', '🔴')  # Critical
                            placeholders['status_placeholder'].write(
                                f"{status_indicator} **Engine ID: {engine_id}**")
                        elif config.OUTPUT_TYPE == 'multiclass':
                            # Get the predicted class label
                            predicted_class_label = classes[y_pred_class]
                            # Map the class label to its status color
                            status_indicator = config.STATUS_COLORS.get(predicted_class_label, '⚪')  # Default to white if not found
                            placeholders['status_placeholder'].write(
                                f"{status_indicator} **Engine ID: {engine_id}**")
                        elif config.OUTPUT_TYPE == 'regression':
                            threshold_value = config.REGRESSION_THRESHOLD
                            if y_pred_value < threshold_value:
                                status_indicator = config.STATUS_COLORS.get('ClassA', '🟢')  # Safe
                            else:
                                status_indicator = config.STATUS_COLORS.get('ClassC', '🔴')  # Critical
                            placeholders['status_placeholder'].write(
                                f"{status_indicator} **Engine ID: {engine_id}**")

                    # **Update Sensor Readings Plot**
                    # Fetch historical sensor data up to current cycle
                    historical_data = engine_data[selected_features].iloc[:cycle_idx + 1]
                    cycles = engine_data['cycle'].iloc[:cycle_idx + 1]

                    fig_sensor = plot_sensor_readings(historical_data, cycles, selected_features, feature_descriptions)
                    placeholders['sensor_placeholder'].plotly_chart(fig_sensor, use_container_width=True,
                                                                    key=f"sensor_{engine_id}_{cycle_idx}")

                    # **Update Gauge or Display Based on Output Type**
                    if config.OUTPUT_TYPE == 'binary':
                        if not np.isnan(y_pred_prob):
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=y_pred_prob * 100,  # Convert to percentage
                                title={'text': f"Cycle {cycle_num}"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#00FF00" if y_pred_prob * 100 < config.BINARY_THRESHOLD * 100 else "#FF0000"},
                                    'steps': [
                                        {'range': [0, config.BINARY_THRESHOLD * 100], 'color': "#00FF00"},
                                        {'range': [config.BINARY_THRESHOLD * 100, 100], 'color': "#FF0000"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': config.BINARY_THRESHOLD * 100
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
                            placeholders['gauge_placeholder'].plotly_chart(fig_gauge,
                                                                           use_container_width=True,
                                                                           key=f"gauge_{engine_id}_{cycle_idx}")
                        else:
                            placeholders['gauge_placeholder'].write(
                                f"Prediction Error for Engine {engine_id}",
                                key=f"gauge_error_{engine_id}_{cycle_idx}")
                    elif config.OUTPUT_TYPE == 'multiclass':
                        # Multiclass Classification: Use Bar Chart
                        y_pred_probs_full = y_pred_probs  # Already obtained above

                        # Validate that classes and y_pred_probs_full have the same length
                        if len(classes) != len(y_pred_probs_full):
                            st.error("Mismatch between number of classes and predicted probabilities.")
                            st.stop()

                        fig_bar = px.bar(
                            x=classes,
                            y=y_pred_probs_full * 100,  # Convert to percentage
                            labels={'x': 'Classes', 'y': 'Probability (%)'},
                            title=f"Class Probabilities at Cycle {cycle_num}",
                            text=y_pred_probs_full * 100
                        )
                        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                        fig_bar.update_layout(yaxis=dict(range=[0, 100]), showlegend=False, height=250)

                        placeholders['gauge_placeholder'].plotly_chart(fig_bar, use_container_width=True,
                                                                        key=f"bar_multiclass_{engine_id}_{cycle_idx}")

                    elif config.OUTPUT_TYPE == 'regression':
                        # Regression: Display predicted value
                        placeholders['gauge_placeholder'].write(
                            f"Predicted Value at Cycle {cycle_num}: {y_pred_value:.2f}",
                            key=f"value_{engine_id}_{cycle_idx}"
                        )

                    # **Update Final Prediction Graph with Unique Key**
                    pred_df = pred_dfs[engine_id]
                    if config.OUTPUT_TYPE == 'binary':
                        y_column = 'Predicted_Probability'
                        y_title = 'Predicted Probability (%)'
                        fig_pred = px.line(
                            pred_df,
                            x='Cycle',
                            y=y_column,
                            title='',
                            height=250
                        )
                        fig_pred.add_hline(y=config.BINARY_THRESHOLD * 100, line_dash='dash', line_color='red',
                                           annotation_text=f'Threshold ({config.BINARY_THRESHOLD * 100}%)')
                        fig_pred.update_yaxes(title_text=y_title)
                        fig_pred.update_xaxes(title_text='Cycle')
                        fig_pred.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                        placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                            key=f"prediction_{engine_id}_{cycle_idx}")
                    elif config.OUTPUT_TYPE == 'multiclass':
                        # Plot probabilities for all classes
                        y_columns = [f'Probability_{cls}' for cls in classes]
                        y_title = 'Predicted Probability (%)'
                        fig_pred = px.line(
                            pred_df,
                            x='Cycle',
                            y=y_columns,
                            title='',
                            height=250
                        )
                        fig_pred.update_yaxes(title_text=y_title)
                        fig_pred.update_xaxes(title_text='Cycle')
                        fig_pred.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                        placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                            key=f"prediction_multiclass_{engine_id}_{cycle_idx}")
                    elif config.OUTPUT_TYPE == 'regression':
                        y_column = 'Predicted_Value'
                        y_title = 'Predicted Value'
                        fig_pred = px.line(
                            pred_df,
                            x='Cycle',
                            y=y_column,
                            title=f"Predicted Values Over Cycles for Engine {engine_id}",
                            height=250
                        )
                        fig_pred.update_yaxes(title_text=y_title)
                        fig_pred.update_xaxes(title_text='Cycle')
                        fig_pred.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                        placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                            key=f"prediction_regression_{engine_id}_{cycle_idx}")

            # Optional: After simulation ends, provide summary or reset button
            if cycle_idx == max_cycles - 1:
                st.session_state['run_multiple_simulation'] = False
                st.success("✅ Multiple Engines Simulation completed.")

def prediction_page(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """Display the prediction page with simulation options."""
    st.header("Simulate Streaming Data and Predict Over Time")
    # Use st.radio() to create tab-like selection
    selected_tab = st.radio(
        "Select Simulation Type",
        ["Single Engine Simulation", "Multiple Engines Simulation"],
        key='prediction_tab_selection'
    )

    if selected_tab == "Single Engine Simulation":
        run_single_engine_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features)
    elif selected_tab == "Multiple Engines Simulation":
        run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features)
    else:
        st.write("Please select a simulation type.")

# ---------------------------
# Main Function
# ---------------------------

def main():
    st.title("Predictive Maintenance using LSTM")
    st.write("""
        **Objective**: Predict if an engine will fail within a certain number of cycles using LSTM neural networks.
    """)

    # Load sensor metadata
    sensors_df = load_csv('../Dataset/sensors.csv')

    # Create mappings for sensor descriptions and abbreviations
    sensor_descriptions = dict(zip(sensors_df['Sensor Name'], sensors_df['Description']))
    sensor_abbreviations = dict(zip(sensors_df['Sensor Name'], sensors_df['Abbreviation']))

    # Load motor specifications
    motors_df = load_csv('../Dataset/motor_specifications.csv')

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Configuration", "Model Training",
                                         "Model Evaluation", "Prediction"])

    # Load or initialize configuration
    config = load_or_initialize_config()

    # ---------------------------
    # Load Existing Configuration
    # ---------------------------
    CONFIG_FILE = '_config.json'
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
            config.update_from_dict(config_data)

    # Ensure essential paths are set
    config.DATASET_PATH = '../Dataset/'
    config.OUTPUT_PATH = '../Output/'

    # Extract parameters from _config
    dataset_path = config.DATASET_PATH
    sequence_length = config.SEQUENCE_LENGTH
    w1 = config.W1
    w0 = config.W0

    @st.cache_data()
    def load_data(_config):
        """Load data using DataLoader."""
        data_loader = DataLoader(_config)
        data_loader.read_data()
        data_loader.output_column = _config.OUTPUT_COLUMN
        return data_loader

    if options != "Introduction":
        data_loader = load_data(config)
        train_df = data_loader.get_train_data()
        test_df = data_loader.get_test_data()
        sequence_cols = data_loader.get_sequence_cols()
        nb_features = data_loader.get_nb_features()
        # Use the sequence length from data_loader to ensure consistency
        sequence_length = data_loader.get_sequence_length()

        # Assign motors to engines
        train_df, test_df, engine_motor_mapping = assign_motors_to_engines(train_df, test_df, motors_df)

    # Navigate to the selected page
    if options == "Introduction":
        introduction()
    elif options == "Data Exploration":
        data_exploration(config, train_df, test_df, motors_df, engine_motor_mapping, sensors_df)
    elif options == "Model Configuration":
        model_configuration(config)
    elif options == "Model Training":
        model_training(config, train_df, sequence_cols, nb_features)
    elif options == "Model Evaluation":
        model_evaluation(config, test_df, sequence_cols, nb_features)
    elif options == "Prediction":
        prediction_page(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features)
    else:
        st.write("Please select an option from the sidebar.")


if __name__ == "__main__":
    main()
