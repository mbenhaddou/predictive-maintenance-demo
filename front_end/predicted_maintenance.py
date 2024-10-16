import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error, roc_auc_score)
from sklearn.preprocessing import label_binarize
from src.lstm.predictive_maintenance import SequenceGenerator, PredictiveMaintenanceModel, DataLoader, generate_test_sequences
from src.lstm.callbacks import StreamlitCallback
from front_end.helprs import *



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
