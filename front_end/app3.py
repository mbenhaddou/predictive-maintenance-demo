# predictive_maintenance_app.py

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import random
import os
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score

sys.path.append('../')
st.set_page_config(layout='wide')


from src.lstm.binary_classification import DataLoader, SequenceGenerator, PredictiveMaintenanceModel, Config, \
    plot_history, evaluate_performance, generate_test_sequences, plot_predictions
from src.lstm.callbacks import StreamlitCallback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.title("Predictive Maintenance using LSTM")
    st.write("""
        **Objective**: Predict if an engine will fail within a certain number of cycles using LSTM neural networks.
    """)

    # Load sensor metadata
    @st.cache_data()
    def load_sensor_metadata():
        sensors_df = pd.read_csv('../Dataset/sensors.csv')
        return sensors_df

    sensors_df = load_sensor_metadata()

    # Create mappings for sensor descriptions and abbreviations
    sensor_descriptions = dict(zip(sensors_df['Sensor Name'], sensors_df['Description']))
    sensor_abbreviations = dict(zip(sensors_df['Sensor Name'], sensors_df['Abbreviation']))

    # Load motor specifications
    @st.cache_data()
    def load_motor_specifications():
        motors_df = pd.read_csv('../Dataset/motor_specifications.csv')
        return motors_df

    motors_df = load_motor_specifications()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Configuration", "Model Training",
                                         "Model Evaluation", "Prediction"])

    # Initialize Config with default values
    config = Config()

    # Config UI Section
    if options == "Model Configuration":
        st.header("Model Configuration")
        st.write("Adjust model parameters below:")
        # Model parameters
        config.DATASET_PATH = st.text_input("Dataset Path", value=config.DATASET_PATH)
        config.SEQUENCE_LENGTH = st.number_input("Sequence Length", min_value=10, max_value=100,
                                                 value=config.SEQUENCE_LENGTH, step=10)
        config.W1 = st.number_input("Threshold W1 for Label Generation", min_value=1, max_value=100, value=config.W1,
                                    step=5)
        config.W0 = st.number_input("Threshold W0 for Label Generation", min_value=1, max_value=100, value=config.W0,
                                    step=5)
        config.LSTM_UNITS = [
            st.number_input("LSTM Units Layer 1", min_value=32, max_value=256, value=config.LSTM_UNITS[0], step=32),
            st.number_input("LSTM Units Layer 2", min_value=32, max_value=256, value=config.LSTM_UNITS[1], step=32)
        ]
        config.DROPOUT_RATES = [
            st.slider("Dropout Rate Layer 1", min_value=0.0, max_value=0.5, value=config.DROPOUT_RATES[0], step=0.1),
            st.slider("Dropout Rate Layer 2", min_value=0.0, max_value=0.5, value=config.DROPOUT_RATES[1], step=0.1)
        ]
        config.L2_REG = st.number_input("L2 Regularization", min_value=0.0, max_value=0.01, value=config.L2_REG,
                                        step=0.001, format="%.3f")
        config.OPTIMIZER = st.selectbox("Optimizer", ["adam", "sgd"], index=0 if config.OPTIMIZER == 'adam' else 1)
        config.LEARNING_RATE = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01,
                                               value=config.LEARNING_RATE, step=0.0001, format="%.4f")
        config.EPOCHS = st.number_input("Epochs", min_value=10, max_value=200, value=config.EPOCHS, step=10)
        config.BATCH_SIZE = st.number_input("Batch Size", min_value=64, max_value=512, value=config.BATCH_SIZE, step=64)

        if config.W1 <= config.W0:
            st.error("W1 should be greater than W0 for proper label generation.")

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

        # Assign motors to engines
        def assign_motors_to_engines(train_df, test_df, motors_df):
            # Get unique motor IDs (Manufacturer Part Numbers)
            motor_ids = motors_df['Manufacturer Part Number'].unique()

            # Combine engine IDs from train and test datasets
            engine_ids = pd.concat([train_df['id'], test_df['id']]).unique()

            # Set seed for reproducibility
            random.seed(41)

            # Assign a random motor to each engine ID
            engine_motor_mapping = {engine_id: random.choice(motor_ids) for engine_id in engine_ids}

            # Map the assigned motors back to the datasets
            train_df['Motor_ID'] = train_df['id'].map(engine_motor_mapping)
            test_df['Motor_ID'] = test_df['id'].map(engine_motor_mapping)

            return train_df, test_df, engine_motor_mapping

        train_df, test_df, engine_motor_mapping = assign_motors_to_engines(train_df, test_df, motors_df)

    if options == "Introduction":
        st.header("Introduction")
        st.write("""
            This application demonstrates the use of LSTM neural networks for predictive maintenance.
            By analyzing historical engine data, the model predicts potential failures before they occur.
        """)
        st.sidebar.info("Use the navigation menu to explore different functionalities of the app.")
        # Optionally, add an image if you have one
        # st.image("engine_image.jpg", use_column_width=True)  # Ensure the image is in the correct path.

    elif options == "Data Exploration":
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
            # Get the motor name for the image filename
            motor_name = motor_spec['Brand']
            # Try different image extensions
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

        st.write(engine_data.head())

        st.subheader("Time Series Plot of Sensors for Selected Engine")
        # Update sensor options with descriptions
        sensor_options = [col for col in train_df.columns if col in sensors_df['Sensor Name'].values]
        sensor_display_options = [f"{sensor} - {sensor_descriptions.get(sensor, '')}" for sensor in sensor_options]

        selected_sensors_display = st.multiselect(
            "Select Sensors to Plot",
            options=sensor_display_options,
            default=sensor_display_options[:3]
        )

        selected_sensors = [option.split(' - ')[0] for option in selected_sensors_display]

        # Time Series Plot using Plotly
        fig = px.line(engine_data, x='cycle', y=selected_sensors,
                      labels={'value': 'Sensor Readings', 'variable': 'Sensor'},
                      title='Sensor Readings Over Time')
        # Update labels with sensor descriptions
        for trace in fig.data:
            sensor_name = trace.name
            trace.name = sensor_descriptions.get(sensor_name, sensor_name)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution of Sensor Readings")
        selected_sensor_hist_option = st.selectbox("Select Sensor for Histogram", sensor_display_options)
        selected_sensor_hist = selected_sensor_hist_option.split(' - ')[0]

        fig_hist = px.histogram(train_df, x=selected_sensor_hist, nbins=50,
                                title=f"Distribution of {sensor_descriptions.get(selected_sensor_hist, selected_sensor_hist)}")
        fig_hist.update_layout(
            xaxis_title=f"{sensor_descriptions.get(selected_sensor_hist, selected_sensor_hist)} ({sensor_abbreviations.get(selected_sensor_hist, '')})",
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

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

            # Accuracy Plot
            fig_acc = px.line(
                history.history,
                y=['accuracy', 'val_accuracy'],
                labels={'index': 'Epoch', 'value': 'Accuracy', 'variable': 'Dataset'},
                title='Model Accuracy Over Epochs'
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Loss Plot
            fig_loss = px.line(
                history.history,
                y=['loss', 'val_loss'],
                labels={'index': 'Epoch', 'value': 'Loss', 'variable': 'Dataset'},
                title='Model Loss Over Epochs'
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    elif options == "Model Evaluation":
        st.header("Model Evaluation")
        st.write("Evaluating the model on test data.")

        # Generate sequences and labels for test data
        seq_array_test, label_array_test = generate_test_sequences(test_df, config.SEQUENCE_LENGTH, sequence_cols)

        pm_model = PredictiveMaintenanceModel(config, nb_features)
        pm_model.load_model()

        # Evaluate on test data
        scores_test = pm_model.evaluate_model(seq_array_test, label_array_test, batch_size=config.BATCH_SIZE)
        st.write(f"Test Accuracy: {scores_test[1] * 100:.2f}%")

        # Predictions on test data
        y_pred_probs = pm_model.predict(seq_array_test)
        y_pred_test = (y_pred_probs > 0.5).astype(int)
        y_true_test = label_array_test

        # Metrics for test data
        st.subheader("Performance Metrics")
        cm = confusion_matrix(y_true_test, y_pred_test)
        st.write(f"Confusion Matrix:")

        # Confusion Matrix Heatmap
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

        precision = precision_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        roc_auc = roc_auc_score(y_true_test, y_pred_probs)

        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {f1:.2f}")
        st.write(f"ROC AUC Score: {roc_auc:.2f}")

        # Plot predictions vs actual
        st.subheader("Predictions vs Actual")
        fig_pred = px.line(
            x=np.arange(len(y_true_test)),
            y=[y_true_test, y_pred_probs.flatten()],
            labels={'x': 'Sample', 'value': 'Label', 'variable': 'Legend'},
            title='Actual vs Predicted Probabilities'
        )
        fig_pred.data[0].name = 'Actual'
        fig_pred.data[1].name = 'Predicted Probability'
        st.plotly_chart(fig_pred, use_container_width=True)






    elif options == "Prediction":

        st.header("Simulate Streaming Data and Predict Over Time")

        # Use st.radio() to create tab-like selection

        selected_tab = st.radio(

            "Select Simulation Type",

            ["Single Engine Simulation", "Multiple Engines Simulation"],

            key='prediction_tab_selection'

        )

        # Initialize session state variables

        if 'run_single_simulation' not in st.session_state:
            st.session_state['run_single_simulation'] = False

        if 'run_multiple_simulation' not in st.session_state:
            st.session_state['run_multiple_simulation'] = False

        if selected_tab == "Single Engine Simulation":

            st.session_state['active_tab'] = "Single Engine Simulation"

            st.session_state['run_multiple_simulation'] = False  # Stop the other simulation

            st.subheader("Single Engine Simulation")

            st.write(

                "Select an engine from the test set to simulate streaming data and generate predictions at each cycle."

            )

            # Select an engine ID from the test set

            engine_ids = test_df['id'].unique()

            selected_id = st.selectbox("Engine ID", engine_ids, key='single_engine_selectbox')

            # Get data for the selected engine

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

                # Get the motor name for the image filename

                motor_name = motor_spec['Brand']

                # Try different image extensions

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

                # Prepare feature descriptions

                all_features = ['fuel_flow', 'motor_speed', 'fan_speed'] + list(sensors_df['Sensor Name'])

                feature_descriptions = {

                    **{feat: feat.replace('_', ' ').title() for feat in ['fuel_flow', 'motor_speed', 'fan_speed']},

                    **{row['Sensor Name']: f"{row['Sensor Name']} - {row['Description']}" for _, row in
                       sensors_df.iterrows()}

                }

                # Allow users to select features to visualize

                selected_features = st.multiselect(

                    "Select Features to Visualize",

                    options=all_features,

                    default=['fuel_flow', 's1', 's2'],

                    format_func=lambda x: feature_descriptions.get(x, x),

                    help="Choose a subset of features to display in real-time.",

                    key='single_engine_features'

                )

                if not selected_features:

                    st.warning("Please select at least one feature to visualize.")

                else:

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

                    start_button = st.button("Start", key='start_single_simulation')

                    stop_button = st.button("Stop", key='stop_single_simulation')

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

                        y_pred_probs = []

                        historical_data = pd.DataFrame(columns=['Cycle'] + selected_features)

                        # Load the model

                        try:

                            pm_model = PredictiveMaintenanceModel(config, nb_features)

                            pm_model.load_model()

                        except Exception as e:

                            st.error(f"Error loading model: {e}")

                            st.stop()

                        for i, (seq, label) in enumerate(zip(sequences, labels)):

                            # Check if the simulation should continue

                            if not st.session_state['run_single_simulation'] or st.session_state[
                                'prediction_tab_selection'] != "Single Engine Simulation":
                                st.write("Simulation stopped.")

                                break

                            # Reshape the sequence to match model input

                            seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)

                            # Make Prediction

                            try:

                                y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]

                            except Exception as e:

                                st.error(f"Error during prediction at sequence {i + 1}: {e}")

                                y_pred_prob = np.nan

                            # Determine the Corresponding Cycle Number

                            if i < len(engine_data):

                                cycle_num = engine_data['cycle'].iloc[i]

                            else:

                                cycle_num = None

                            # Append to Lists

                            cycle_list.append(cycle_num)

                            y_pred_probs.append(y_pred_prob)

                            # Update Historical Sensor Readings

                            if cycle_num is not None:
                                current_features = seq[-1][[sequence_cols.index(feat) for feat in selected_features]]

                                input_data = {feat: val for feat, val in zip(selected_features, current_features)}

                                input_data['Cycle'] = cycle_num

                                historical_data = historical_data._append(input_data, ignore_index=True)

                            # Update Input Visualization with Historical Data (Sliding Window)

                            with input_placeholder.container():

                                st.write(f"**Cycle {cycle_num}**")

                                if not historical_data.empty:

                                    windowed_data = historical_data.tail(50)

                                    fig_input = go.Figure()

                                    for feature in selected_features:
                                        feature_label = feature_descriptions.get(feature, feature)

                                        fig_input.add_trace(go.Scatter(

                                            x=windowed_data['Cycle'],

                                            y=windowed_data[feature],

                                            mode='lines+markers',

                                            name=feature_label

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

                            # Update Output Visualization using Plotly Gauge

                            with output_placeholder.container():

                                if not np.isnan(y_pred_prob):

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

                                    fig_gauge.update_layout(width=600, height=400)

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

                        fig_pred = px.line(

                            pred_df,

                            x='Cycle',

                            y='Predicted_Probability_of_Failure',

                            title=f"Predicted Probability of Failure Over Cycles for Engine {selected_id}"

                        )

                        fig_pred.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='Threshold (50%)')

                        fig_pred.update_yaxes(title_text='Predicted Probability (%)')

                        fig_pred.update_xaxes(title_text='Cycle')

                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Optionally, Allow Users to Download the Predictions

                        csv = pred_df.to_csv(index=False).encode('utf-8')

                        st.download_button(

                            label="Download Predictions as CSV",

                            data=csv,

                            file_name='predictions.csv',

                            mime='text/csv',

                        )

                    else:

                        st.write("Click the 'Start Simulation' button to begin.")

        elif selected_tab == "Multiple Engines Simulation":
            st.session_state['active_tab'] = "Multiple Engines Simulation"
            st.session_state['run_single_simulation'] = False  # Stop the other simulation

            st.subheader("Multiple Engines Simulation")
            st.write(
                "Simulate streaming data and predictions for multiple engines simultaneously."
            )

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
                # Randomly select engines
                selected_ids = random.sample(list(test_df['id'].unique()), num_engines)
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
            else:
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
                start_button = st.button("Start", key='start_multiple_simulation')
                stop_button = st.button("Stop", key='stop_multiple_simulation')

                if start_button:
                    st.session_state['run_multiple_simulation'] = True
                if stop_button:
                    st.session_state['run_multiple_simulation'] = False

                if st.session_state['run_multiple_simulation']:
                    # Load the model
                    try:
                        pm_model = PredictiveMaintenanceModel(config, nb_features)
                        pm_model.load_model()
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        st.stop()

                    # Prepare data for selected engines
                    engines_data = {}
                    sequences_dict = {}
                    labels_dict = {}
                    max_cycles = 0

                    for engine_id in selected_ids:
                        engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
                        # Generate sequences
                        seq_gen = SequenceGenerator(
                            df=engine_data,
                            sequence_length=config.SEQUENCE_LENGTH,
                            sequence_cols=sequence_cols
                        )
                        sequences, labels = seq_gen.generate_sequences()
                        sequences_dict[engine_id] = sequences
                        labels_dict[engine_id] = labels
                        engines_data[engine_id] = engine_data
                        max_cycles = max(max_cycles, len(sequences))

                    # Initialize DataFrames to Store Predictions
                    pred_dfs = {engine_id: pd.DataFrame(columns=['Cycle', 'Predicted_Probability_of_Failure']) for
                                engine_id in selected_ids}

                    # Initialize placeholders for each engine
                    engine_placeholders = {}
                    for engine_id in selected_ids:
                        # For each engine, create a container (row)
                        engine_container = st.container()
                        with engine_container:
                            # Create four columns for this engine
                            cols = st.columns([0.5, 2, 0.8, 2])  # Adjusted column widths

                            # Column 1: Status emoji and Engine ID
                            with cols[0]:
                                status_placeholder = st.empty()
                                # Initial display with green circle
                                status_indicator = '🟢'
                                status_placeholder.write(f"{status_indicator} **Engine ID: {engine_id}**")

                            # Placeholders for the plots in columns 2, 3, 4
                            sensor_placeholder = cols[1].empty()
                            gauge_placeholder = cols[2].empty()
                            prediction_placeholder = cols[3].empty()
                            # Store the placeholders
                            engine_placeholders[engine_id] = {
                                'status_placeholder': status_placeholder,
                                'sensor_placeholder': sensor_placeholder,
                                'gauge_placeholder': gauge_placeholder,
                                'prediction_placeholder': prediction_placeholder,
                                'container': engine_container,
                                'cols': cols
                            }

                    # Simulation loop
                    for cycle_idx in range(max_cycles):
                        # Check if the simulation should continue
                        if not st.session_state['run_multiple_simulation'] or st.session_state[
                            'prediction_tab_selection'] != "Multiple Engines Simulation":
                            st.write("Simulation stopped.")
                            break

                        # For each engine, update the displays
                        for engine_id in selected_ids:
                            sequences = sequences_dict[engine_id]
                            engine_data = engines_data[engine_id]
                            placeholders = engine_placeholders[engine_id]
                            if cycle_idx < len(sequences):
                                seq = sequences[cycle_idx]
                                # Reshape sequence
                                seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)
                                # Make prediction
                                try:
                                    y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
                                except Exception as e:
                                    st.error(
                                        f"Error during prediction for engine {engine_id} at cycle {cycle_idx + 1}: {e}")
                                    y_pred_prob = np.nan

                                # Get cycle number
                                cycle_num = engine_data['cycle'].iloc[cycle_idx]

                                # Append prediction to DataFrame
                                pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                                    'Cycle': cycle_num,
                                    'Predicted_Probability_of_Failure': y_pred_prob
                                }, ignore_index=True)

                                # Update status indicator based on prediction
                                with placeholders['status_placeholder']:
                                    status_indicator = '🟢' if y_pred_prob * 100 < 50 else '🔴'  # Threshold at 50%
                                    placeholders['status_placeholder'].write(
                                        f"{status_indicator} **Engine ID: {engine_id}**")

                                # Update sensor readings plot
                                # Select a few sensors to display (adjust as needed)
                                selected_features = ['s1', 's2', 's3']

                                current_features = engine_data[selected_features].iloc[:cycle_idx + 1]
                                cycles = engine_data['cycle'].iloc[:cycle_idx + 1]

                                # Create sensor readings plot
                                fig_sensor = go.Figure()
                                for feature in selected_features:
                                    fig_sensor.add_trace(go.Scatter(
                                        x=cycles,
                                        y=current_features[feature],
                                        mode='lines',
                                        name=feature
                                    ))
                                fig_sensor.update_layout(
                                    xaxis_title='Cycle',
                                    yaxis_title='Sensor Readings',
                                    showlegend=True,
                                    height=250,
                                    margin=dict(l=0, r=0, t=20, b=0)
                                )
                                # Update sensor readings placeholder
                                placeholders['sensor_placeholder'].plotly_chart(fig_sensor, use_container_width=True)

                                # Update gauge
                                if not np.isnan(y_pred_prob):
                                    fig_gauge = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=y_pred_prob * 100,  # Convert to percentage
                                        title={'text': f"Cycle {cycle_num}"},
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
                                    fig_gauge.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
                                    placeholders['gauge_placeholder'].plotly_chart(fig_gauge, use_container_width=True)
                                else:
                                    placeholders['gauge_placeholder'].write(f"Prediction Error for Engine {engine_id}")

                                # Update final prediction graph
                                # Plot the predictions up to current cycle
                                pred_df = pred_dfs[engine_id]
                                fig_pred = px.line(
                                    pred_df,
                                    x='Cycle',
                                    y='Predicted_Probability_of_Failure',
                                    title='',
                                    height=250
                                )
                                fig_pred.add_hline(y=0.5, line_dash='dash', line_color='red',
                                                   annotation_text='Threshold (50%)')
                                fig_pred.update_yaxes(title_text='Predicted Probability (%)')
                                fig_pred.update_xaxes(title_text='Cycle')
                                fig_pred.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                                placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True)
                            else:
                                # Engine has completed all cycles
                                pass

                        # Simulate real-time by adding a delay
                        time.sleep(speed)

                else:
                    st.write("Click the 'Start Simulation' button to begin.")


    else:
        st.write("Please select an option from the sidebar.")


if __name__ == "__main__":
    main()
