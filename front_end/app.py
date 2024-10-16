# predictive_maintenance_app.py

import streamlit as st
import numpy as np
import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
sys.path.append('../')

from src.lstm.predictive_maintenance import DataLoader, SequenceGenerator, PredictiveMaintenanceModel, Config, plot_history, evaluate_performance, generate_test_sequences, plot_predictions
from src.lstm.callbacks import StreamlitCallback
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Predictive Maintenance using LSTM")
    st.write("""
        **Objective**: Predict if an engine will fail within a certain number of cycles using LSTM neural networks.
    """)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Configuration", "Model Training", "Model Evaluation", "Prediction"])

    # Initialize Config with default values
    config = Config()

    # Config UI Section
    if options == "Model Configuration":
        st.header("Model Configuration")
        st.write("Adjust model parameters below:")
        # Model parameters
        config.DATASET_PATH = st.text_input("Dataset Path", value=config.DATASET_PATH)
        config.SEQUENCE_LENGTH = st.number_input("Sequence Length", min_value=10, max_value=100, value=config.SEQUENCE_LENGTH, step=10)
        config.W1 = st.number_input("Threshold W1 for Label Generation", min_value=1, max_value=100, value=config.W1, step=5)
        config.W0 = st.number_input("Threshold W0 for Label Generation", min_value=1, max_value=100, value=config.W0, step=5)
        config.LSTM_UNITS = [
            st.number_input("LSTM Units Layer 1", min_value=32, max_value=256, value=config.LSTM_UNITS[0], step=32),
            st.number_input("LSTM Units Layer 2", min_value=32, max_value=256, value=config.LSTM_UNITS[1], step=32)
        ]
        config.DROPOUT_RATES = [
            st.slider("Dropout Rate Layer 1", min_value=0.0, max_value=0.5, value=config.DROPOUT_RATES[0], step=0.1),
            st.slider("Dropout Rate Layer 2", min_value=0.0, max_value=0.5, value=config.DROPOUT_RATES[1], step=0.1)
        ]
        config.L2_REG = st.number_input("L2 Regularization", min_value=0.0, max_value=0.01, value=config.L2_REG, step=0.001, format="%.3f")
        config.OPTIMIZER = st.selectbox("Optimizer", ["adam", "sgd"], index=0 if config.OPTIMIZER == 'adam' else 1)
        config.LEARNING_RATE = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=config.LEARNING_RATE, step=0.0001, format="%.4f")
        config.EPOCHS = st.number_input("Epochs", min_value=10, max_value=200, value=config.EPOCHS, step=10)
        config.BATCH_SIZE = st.number_input("Batch Size", min_value=64, max_value=512, value=config.BATCH_SIZE, step=64)

        if st.button("Save Configuration"):
            st.success("Configuration saved.")

    # Extract parameters from config
    dataset_path = config.DATASET_PATH
    sequence_length = config.SEQUENCE_LENGTH
    w1 = config.W1
    w0 = config.W0

    @st.cache_data()
    def load_data(dataset_path, sequence_length, w1, w0):
        data_loader = DataLoader(
            dataset_path=dataset_path,
            sequence_length=sequence_length,
            w1=w1,
            w0=w0
        )
        data_loader.read_data()
        return data_loader

    if options != "Introduction":
        data_loader = load_data(dataset_path, sequence_length, w1, w0)
        train_df = data_loader.get_train_data()
        test_df = data_loader.get_test_data()
        sequence_cols = data_loader.get_sequence_cols()
        nb_features = data_loader.get_nb_features()
        # Use the sequence length from data_loader to ensure consistency
        sequence_length = data_loader.get_sequence_length()

    if options == "Introduction":
        st.header("Introduction")
        st.write("""
            This application demonstrates the use of LSTM neural networks for predictive maintenance.
            By analyzing historical engine data, the model predicts potential failures before they occur.
        """)
        # Optionally, add an image if you have one
        # st.image("engine_image.jpg", use_column_width=True)  # Ensure the image is in the correct path.

    elif options == "Data Exploration":
        st.header("Data Exploration")
        st.subheader("Select an Engine ID to Explore")
        engine_ids = train_df['id'].unique()
        selected_id = st.selectbox("Engine ID", engine_ids)

        # Filter data for the selected engine ID
        engine_data = train_df[train_df['id'] == selected_id]

        st.write(f"Showing data for Engine ID: {selected_id}")
        st.write(engine_data.head())

        st.subheader("Time Series Plot of Sensors for Selected Engine")
        # Select sensors to plot
        sensor_options = [col for col in train_df.columns if col.startswith('s')]
        selected_sensors = st.multiselect("Select Sensors to Plot", sensor_options, default=sensor_options[:3])

        fig, ax = plt.subplots(figsize=(10, 6))
        for sensor in selected_sensors:
            ax.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Sensor Readings')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Distribution of Sensor Readings")
        selected_sensor_hist = st.selectbox("Select Sensor for Histogram", sensor_options)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(train_df[selected_sensor_hist], bins=50, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel('Sensor Reading')
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)

    elif options == "Model Training":
        st.header("Model Training")
        st.write("Training the model with the configured parameters.")

        # Generate sequences and labels for training
        seq_gen = SequenceGenerator(train_df, config.SEQUENCE_LENGTH, sequence_cols)
        seq_array, label_array = seq_gen.generate_sequences()

        if st.button("Train Model"):
            pm_model = PredictiveMaintenanceModel(config, nb_features)
            pm_model.build_model()

            # Placeholders for progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create the Streamlit callback
            epochs = config.EPOCHS
            st_callback = StreamlitCallback(epochs, progress_bar, status_text)

            # Train model with the custom callback
            history = pm_model.train_model(
                seq_array, label_array,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                custom_callback=st_callback  # Pass the custom callback
            )
            st.success("Model training completed.")

            # Plot training history
            st.subheader("Training History")
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(history.history['accuracy'], label='Train Accuracy')
            ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()
            st.pyplot(fig_acc)

            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history['loss'], label='Train Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

    elif options == "Model Evaluation":
        st.header("Model Evaluation")
        st.write("Evaluating the model on test data.")

        # Generate sequences and labels for test data
        seq_array_test, label_array_test = generate_test_sequences(test_df, config.SEQUENCE_LENGTH, sequence_cols)

        pm_model = PredictiveMaintenanceModel(config, nb_features)
        pm_model.load_model()

        # Evaluate on test data
        scores_test = pm_model.evaluate_model(seq_array_test, label_array_test, batch_size=config.BATCH_SIZE)
        st.write(f"Test Accuracy: {scores_test[1]*100:.2f}%")

        # Predictions on test data
        y_pred_test = pm_model.predict(seq_array_test)
        y_true_test = label_array_test

        # Metrics for test data
        st.subheader("Performance Metrics")
        cm = confusion_matrix(y_true_test, y_pred_test)
        st.write(f"Confusion Matrix:")
        st.write(cm)
        precision = precision_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {f1:.2f}")

        # Plot predictions vs actual
        st.subheader("Predictions vs Actual")
        fig_pred, ax_pred = plt.subplots()
        ax_pred.plot(y_pred_test, label='Predicted', linestyle='--')
        ax_pred.plot(y_true_test, label='Actual', alpha=0.7)
        ax_pred.set_xlabel('Sample')
        ax_pred.set_ylabel('Label')
        ax_pred.legend()
        st.pyplot(fig_pred)

    elif options == "Prediction":
        st.header("Simulate Streaming Data and Predict Over Time")
        st.write(
            "Select an engine from the test set to simulate streaming data and generate predictions at each cycle."
        )

        # Select an engine ID from the test set
        engine_ids = test_df['id'].unique()
        selected_id = st.selectbox("Engine ID", engine_ids)

        # Get data for the selected engine
        engine_data = test_df[test_df['id'] == selected_id].reset_index(drop=True)

        st.write(f"Selected Engine ID: {selected_id}")
        st.write("### Engine Data Preview:")
        st.dataframe(engine_data.head())

        # Validate Sequence Columns
        missing_cols = [col for col in sequence_cols if col not in engine_data.columns]
        if missing_cols:
            st.error(f"The following sequence columns are missing in the uploaded data: {missing_cols}")
        else:
            # Generate Sequences with Padding using SequenceGenerator
            try:
                seq_gen = SequenceGenerator(
                    df=engine_data,
                    sequence_length=config.SEQUENCE_LENGTH,
                    sequence_cols=sequence_cols
                )
                sequences, labels = seq_gen.generate_sequences()
                total_sequences = len(sequences)
                st.success(f"Generated {total_sequences} sequences for prediction.")
            except Exception as e:
                st.error(f"Error in generating sequences: {e}")
                st.stop()

            # Allow users to select features to visualize
            sensor_options = [col for col in sequence_cols if col.startswith('s')]
            selected_features = st.multiselect(
                "Select Features to Visualize",
                options=['voltage_input', 'current_limit', 'speed_control'] + sensor_options,
                default=['voltage_input', 's1', 's2'],
                help="Choose a subset of features to display in real-time."
            )

            if not selected_features:
                st.warning("Please select at least one feature to visualize.")
            else:
                # Initialize Lists to Store Predictions
                cycle_list = []
                y_pred_probs = []

                # Initialize DataFrame to Store Historical Sensor Readings
                historical_data = pd.DataFrame(columns=['Cycle'] + selected_features)
                # **New Feature**: Configurable Sliding Window for Sensor Readings
                sensor_window_size = st.number_input(
                    "Sensor Readings Window Size",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of latest sensor readings to display in the graph."
                )

                # Initialize Placeholders for Real-Time Visualization
                input_placeholder = st.empty()
                output_placeholder = st.empty()

                # Optionally, add a speed slider to control the simulation speed
                speed = st.slider(
                    "Select Simulation Speed (Seconds per Cycle)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="Control how fast the simulation runs."
                )

                # Load the model
                try:
                    pm_model = PredictiveMaintenanceModel(config, nb_features)
                    pm_model.load_model()
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()

                # Iterate Over Sequences and Make Predictions with Real-Time Updates
                for i, (seq, label) in enumerate(zip(sequences, labels)):
                    # Reshape the sequence to match model input: (1, sequence_length, num_features)
                    seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)

                    # Make Prediction
                    try:
                        y_pred_prob = pm_model.model.predict(seq_reshaped)[0][
                            0]  # Adjust based on your model's predict method
                    except Exception as e:
                        st.error(f"Error during prediction at sequence {i + 1}: {e}")
                        y_pred_prob = np.nan  # Assign NaN to indicate prediction failure

                    # Determine the Corresponding Cycle Number
                    if i < len(engine_data):
                        cycle_num = engine_data['cycle'].iloc[i]
                    else:
                        cycle_num = None  # Or handle appropriately

                    # Append to Lists
                    cycle_list.append(cycle_num)
                    y_pred_probs.append(y_pred_prob)

                    # Update Historical Sensor Readings
                    if cycle_num is not None:
                        current_features = seq[-1][[sequence_cols.index(feat) for feat in selected_features]]
                        input_data = {feat: val for feat, val in zip(selected_features, current_features)}
                        input_data['Cycle'] = cycle_num
                        historical_data = historical_data._append(input_data, ignore_index=True)

                    # **Update Input Visualization with Historical Data (Sliding Window)**
                    with input_placeholder.container():
                        st.write(f"**Cycle {cycle_num}**")
                        if not historical_data.empty:
                            # Slice the DataFrame to only include the last 'sensor_window_size' entries
                            windowed_data = historical_data.tail(sensor_window_size)

                            fig_input = go.Figure()
                            for feature in selected_features:
                                fig_input.add_trace(go.Scatter(
                                    x=windowed_data['Cycle'],
                                    y=windowed_data[feature],
                                    mode='lines+markers',
                                    name=feature
                                ))
                            fig_input.update_layout(
                                xaxis_title='Cycle',
                                yaxis_title='Sensor Readings',
                                showlegend=True,
                                height=400
                            )
                            st.plotly_chart(fig_input, use_container_width=True)
                        else:
                            st.write("No historical data to display.")

                    # **Update Output Visualization using Plotly's Larger Gauge**
                    with output_placeholder.container():
                        if not np.isnan(y_pred_prob):
                            # Create a larger Plotly Gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=y_pred_prob * 100,  # Convert to percentage
                                title={'text': "Probability of Failure (%)"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#00FF00" if y_pred_prob < 50 else "#FF0000"},
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
                            fig_gauge.update_layout(width=600, height=400)  # Increased size
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        else:
                            st.write("Prediction Error")

                    # Simulate real-time by adding a delay
                    time.sleep(speed)

                # Create a DataFrame for All Predictions
                pred_df = pd.DataFrame({
                    'Cycle': cycle_list,
                    'Predicted_Probability_of_Failure': y_pred_probs
                })

                # Handle NaN Predictions
                pred_df.dropna(inplace=True)

                # Display the Predictions DataFrame
                st.subheader("Prediction Results")
                st.dataframe(pred_df)

                # Plot the Predictions Over Time
                st.subheader(f"Predicted Probability of Failure Over Time for Engine {selected_id}")
                fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                ax_pred.plot(pred_df['Cycle'].values, pred_df['Predicted_Probability_of_Failure'].values,
                             label='Predicted Probability', marker='o')
                ax_pred.axhline(y=50, color='r', linestyle='--', label='Threshold (50%)')
                ax_pred.set_xlabel('Cycle')
                ax_pred.set_ylabel('Predicted Probability (%)')
                ax_pred.set_ylim([0, 1])
                ax_pred.set_title(f"Predicted Probability of Failure Over Cycles for Engine {selected_id}")
                ax_pred.legend()
                ax_pred.grid(True)
                st.pyplot(fig_pred)

                # Optionally, Allow Users to Download the Predictions
                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )


    else:
        st.write("Please select an option from the sidebar.")

if __name__ == "__main__":
    main()
