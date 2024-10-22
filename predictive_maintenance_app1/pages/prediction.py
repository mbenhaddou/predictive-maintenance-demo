# pages/prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import time, os
import plotly.express as px
import plotly.graph_objects as go
import random
from utils.helpers import (
    plot_sensor_readings,
    plot_confusion_matrix,
    plot_roc_curve_binary,
    plot_multiclass_roc,
    plot_regression_predictions,
    initialize_placeholders
)
from model import PredictiveMaintenanceModel, SequenceGenerator


def display(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """Display the Prediction page with simulation options."""
    st.header("Prediction Simulation")
    st.write("""
        Simulate streaming data and observe real-time predictions from the trained LSTM model.
        Choose between simulating a single engine, multiple engines, or all engines on a map view.
    """)

    # Add Map View Simulation as a third option
    selected_tab = st.radio(
        "Select Simulation Type",
        ["Single Engine Simulation", "Multiple Engines Simulation", "Map View Simulation"],
        key='prediction_tab_selection', index=1
    )

    if selected_tab == "Single Engine Simulation":
        run_single_engine_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols,
                                     nb_features)
    elif selected_tab == "Multiple Engines Simulation":
        run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols,
                                        nb_features)
    elif selected_tab == "Map View Simulation":
        run_map_view_simulation(config, test_df, engine_motor_mapping, sequence_cols, nb_features)
    else:
        st.write("Please select a simulation type.")

def run_map_view_simulation(config, test_df, engine_motor_mapping, sequence_cols, nb_features):
    """Run the Map View Simulation displaying a grid of engine statuses with cycle info."""
    st.subheader("Map View Simulation")
    st.write("Simulate all engines simultaneously and display a grid of their statuses.")

    # Simulation Speed Slider
    speed = st.slider(
        "Select Simulation Speed (Seconds per Cycle)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Control how fast the simulation runs.",
        key='map_view_speed'
    )

    # Start/Stop Simulation Buttons
    start_button = st.button("Start Simulation", key='start_map_simulation')
    stop_button = st.button("Stop Simulation", key='stop_map_simulation')

    # Initialize session state for simulation control
    if 'run_map_simulation' not in st.session_state:
        st.session_state['run_map_simulation'] = False

    # Start the simulation when the Start button is pressed
    if start_button:
        st.session_state['run_map_simulation'] = True

    # Stop the simulation when the Stop button is pressed
    if stop_button:
        st.session_state['run_map_simulation'] = False

    if st.session_state['run_map_simulation']:
        # Load model once for all engines
        try:
            pm_model = PredictiveMaintenanceModel(config, nb_features, config.MODEL_TYPE)
            pm_model.load_and_build_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Define classes for multiclass
        classes = config.class_labels if config.MODEL_TYPE == 'multiclass' and hasattr(config, 'class_labels') else [
            'Class_A', 'Class_B', 'Class_C']

        # Prepare data and initialize status tracking for each engine
        engine_ids = test_df['id'].unique()
        num_engines = len(engine_ids)

        # Calculate grid dimensions (e.g., 10x10 for 100 engines)
        grid_size = int(np.ceil(np.sqrt(num_engines)))
        x_coords, y_coords = np.meshgrid(range(grid_size), range(grid_size))
        grid_data = pd.DataFrame(
            {'x': x_coords.flatten()[:num_engines], 'y': y_coords.flatten()[:num_engines], 'id': engine_ids}
        )

        # Placeholders for map, cycle information, and lists
        map_placeholder = st.empty()
        cycle_info_placeholder = st.empty()


        # Generate sequences for all engines and initialize cycles remaining
        sequence_dict = {}
        engine_cycles_remaining = {}
        engines_data = {}
        for engine_id in engine_ids:
            engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
            seq_gen = SequenceGenerator(
                df=engine_data,
                sequence_length=config.SEQUENCE_LENGTH,
                sequence_cols=sequence_cols,
                output_column=config.OUTPUT_COLUMN
            )
            sequences, _ = seq_gen.generate_sequences()
            sequence_dict[engine_id] = sequences
            engine_cycles_remaining[engine_id] = len(sequences)
            engines_data[engine_id] = engine_data

        # Initialize DataFrames to store predictions for each engine
        pred_dfs = {engine_id: pd.DataFrame(
            columns=['Cycle', 'Predicted_Probability'] if config.MODEL_TYPE == 'binary' else (
                    ['Cycle'] + [f'Probability_{cls}' for cls in classes]) if config.MODEL_TYPE == 'multiclass' else ['Cycle', 'Predicted_Value'])
            for engine_id in engine_ids}

        # Initialize placeholders for each engine
        engine_placeholders = {}
        for engine_id in engine_ids:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    status_placeholder = st.empty()
                with col2:
                    gauge_placeholder = st.empty()
                    prediction_placeholder = st.empty()
            engine_placeholders[engine_id] = {
                'status_placeholder': status_placeholder,
                'gauge_placeholder': gauge_placeholder,
                'prediction_placeholder': prediction_placeholder
            }

        # Track engines in critical and warning states
        critical_engines = set()
        warning_engines = set()

        # Simulation Loop
        cycle_idx = 0
        while st.session_state['run_map_simulation']:
            # Check if all engines have completed their cycles
            if all(cycles == 0 for cycles in engine_cycles_remaining.values()):
                st.write("All engines have completed their cycles.")
                break

            # Update cycle information
            cycle_info_placeholder.markdown(f"### Cycle: {cycle_idx + 1}")

            # Collect sequences for current cycle for engines that have data for this cycle
            batch_sequences = []
            engine_sequence_map = {}
            for engine_id, sequences in sequence_dict.items():
                if engine_cycles_remaining[engine_id] > 0 and cycle_idx < len(sequences):
                    batch_sequences.append(sequences[cycle_idx])
                    engine_sequence_map[engine_id] = len(batch_sequences) - 1  # Track index in batch_sequences
                else:
                    continue  # Skip engines with no remaining data

            if not batch_sequences:
                break  # Stop if no sequences left to process

            # Convert to a 3D numpy array for batch prediction
            batch_sequences = np.array(batch_sequences).reshape(len(batch_sequences), config.SEQUENCE_LENGTH, nb_features)

            # Run batch prediction for all engines with data for this cycle
            predictions = pm_model.model.predict(batch_sequences)

            # Update status based on predictions
            engine_status = {}
            for engine_id in engine_ids:
                if engine_cycles_remaining[engine_id] <= 0:
                    engine_status[engine_id] = 'completed'
            for engine_id, index in engine_sequence_map.items():
                prediction = predictions[index]
                status = 'safe'

                # Determine status and extract predicted values
                if config.MODEL_TYPE == 'binary':
                    y_pred_prob = prediction[0]
                    y_pred_class = int(y_pred_prob > config.BINARY_THRESHOLD)
                    status = 'critical' if y_pred_class else 'safe'
                    y_pred_probs = None
                    y_pred_value = None
                elif config.MODEL_TYPE == 'multiclass':
                    y_pred_probs = prediction
                    y_pred_class = np.argmax(y_pred_probs)
                    status = 'critical' if y_pred_class == 2 else 'warning' if y_pred_class == 1 else 'safe'
                    y_pred_prob = None
                    y_pred_value = None
                elif config.MODEL_TYPE == 'regression':
                    y_pred_value = prediction[0]
                    status = 'critical' if y_pred_value > config.REGRESSION_THRESHOLD else 'safe'
                    y_pred_prob = None
                    y_pred_probs = None

                engine_status[engine_id] = status

                # **Updated code to manage critical and warning engines**
                if status == 'critical':
                    critical_engines.add(engine_id)
                    warning_engines.discard(engine_id)  # Remove from warning if present
                elif status == 'warning':
                    warning_engines.add(engine_id)
                    critical_engines.discard(engine_id)  # Remove from critical if present
                else:
                    critical_engines.discard(engine_id)
                    warning_engines.discard(engine_id)

                # Accumulate predictions
                cycle_num = engines_data[engine_id]['cycle'].iloc[cycle_idx] if cycle_idx < len(engines_data[engine_id]) else f"Cycle_{cycle_idx + 1}"

                if config.MODEL_TYPE == 'binary':
                    pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                        'Cycle': cycle_num,
                        'Predicted_Probability': y_pred_prob * 100  # Convert to percentage
                    }, ignore_index=True)
                elif config.MODEL_TYPE == 'multiclass':
                    pred_entry = {'Cycle': cycle_num}
                    for cls_idx, cls in enumerate(classes):
                        pred_entry[f'Probability_{cls}'] = y_pred_probs[cls_idx] * 100  # Convert to percentage
                    pred_dfs[engine_id] = pred_dfs[engine_id]._append(pred_entry, ignore_index=True)
                elif config.MODEL_TYPE == 'regression':
                    pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                        'Cycle': cycle_num,
                        'Predicted_Value': y_pred_value
                    }, ignore_index=True)

                # Decrement the remaining cycles for the engine
                engine_cycles_remaining[engine_id] -= 1

            grid_data['status'] = grid_data['id'].map(engine_status)
            grid_data['color'] = grid_data['status'].map({
                'safe': 'green',
                'warning': 'yellow',
                'critical': 'red',
                'completed': 'gray'
            })
            grid_data['symbol'] = grid_data['status'].map({
                'safe': 'circle',
                'warning': 'circle',
                'critical': 'circle',
                'completed': 'circle-open'
            })

            # Display the status map as a grid of colored circles
            fig = px.scatter(
                grid_data, x='x', y='y',
                color='color',
                symbol='symbol',
                color_discrete_map={
                    'green': 'green',
                    'yellow': 'yellow',
                    'red': 'red',
                    'gray': 'gray'
                },
                symbol_map={
                    'circle': 'circle',
                    'circle-open': 'circle-open'
                },
                labels={'color': 'Status'},
                title='Engine Status Map',
                category_orders={'color': ['green', 'yellow', 'red', 'gray']}
            )
            fig.update_traces(marker=dict(size=15))
            fig.update_layout(
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            # Update map in place with unique key
            map_placeholder.plotly_chart(fig, use_container_width=True, key=f"status_map_{cycle_idx}")

            # Display only critical and warning engines
            if critical_engines:
                #it critical_list_placeholder.subheader("Critical Engines")
                for engine_id in critical_engines:
                    if engine_cycles_remaining[engine_id] >= 0:
                        # Use process_engine_prediction
                        process_engine_prediction(
                            engine_id=engine_id,
                            cycle_idx=cycle_idx,
                            engine_data=engines_data[engine_id],
                            placeholders=engine_placeholders[engine_id],
                            config=config,
                            selected_features=None,
                            feature_descriptions=None,
                            pred_df=pred_dfs[engine_id],
                            classes=classes
                        )

            if warning_engines:
                #warning_list_placeholder.subheader("Warning Engines")
                for engine_id in warning_engines:
                    if engine_cycles_remaining[engine_id] >= 0:
                        process_engine_prediction(
                            engine_id=engine_id,
                            cycle_idx=cycle_idx,
                            engine_data=engines_data[engine_id],
                            placeholders=engine_placeholders[engine_id],
                            config=config,
                            selected_features=None,
                            feature_descriptions=None,
                            pred_df=pred_dfs[engine_id],
                            classes=classes
                        )

            # Simulate real-time by adding a delay
            time.sleep(speed)

            # Increment cycle index
            cycle_idx += 1

        # Reset simulation state if simulation completes or stops
        if not st.session_state['run_map_simulation']:
            st.session_state['run_map_simulation'] = False
            st.success("Simulation stopped.")
        else:
            st.success("All engines have completed their cycles. Simulation completed.")

def run_single_engine_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """
    Handle the Single Engine Simulation.

    Parameters:
    - config: Config object containing model and simulation parameters.
    - test_df: DataFrame containing test data.
    - engine_motor_mapping: Dictionary mapping engine IDs to motor IDs.
    - motors_df: DataFrame containing motor specifications.
    - sensors_df: DataFrame containing sensor metadata.
    - sequence_cols: List of sensor feature column names.
    - nb_features: Number of features in the input sequences.
    """
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
            image_path = f'Dataset/images/{motor_name}{ext}'
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

    if 'run_single_simulation' not in st.session_state:
        st.session_state['run_single_simulation'] = False

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
            pm_model = PredictiveMaintenanceModel(config, nb_features, config.MODEL_TYPE)
            pm_model.load_and_build_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Define the classes list for multiclass classification
        if config.MODEL_TYPE == 'multiclass':
            classes = config.class_labels if hasattr(config, 'class_labels') else ['Class_A', 'Class_B', 'Class_C']
            num_classes = len(classes)
        elif config.MODEL_TYPE == 'binary':
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
                if config.MODEL_TYPE == 'binary':
                    y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
                    y_pred_class = (y_pred_prob > config.BINARY_THRESHOLD).astype(int)
                elif config.MODEL_TYPE == 'multiclass':
                    y_pred_probs = pm_model.model.predict(seq_reshaped)[0]
                    y_pred_class = np.argmax(y_pred_probs)
                    y_pred_prob = y_pred_probs[y_pred_class]
                elif config.MODEL_TYPE == 'regression':
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
            if config.MODEL_TYPE == 'binary':
                y_pred_list.append(y_pred_prob)
            elif config.MODEL_TYPE == 'multiclass':
                y_pred_list.append(y_pred_probs)  # Store all class probabilities
            elif config.MODEL_TYPE == 'regression':
                y_pred_list.append(y_pred_value)

            # Update Historical Sensor Readings
            if cycle_num != f"Cycle_{i + 1}":
                # Assuming 'seq' is a 2D array: (sequence_length, nb_features)
                # Extract the last cycle's features
                current_features = seq[-1][[sequence_cols.index(feat) for feat in selected_features]]
                input_data = {feat: val for feat, val in zip(selected_features, current_features)}
                input_data['Cycle'] = cycle_num
                historical_data = historical_data._append(input_data, ignore_index=True)

            # Update Input Visualization with Historical Data (Sliding Window)
            with input_placeholder.container():
                st.write(f"**Cycle {cycle_num}**")
                if not historical_data.empty:
                    windowed_data = historical_data.tail(50)  # Display last 50 cycles
                    fig_input = plot_sensor_readings(windowed_data[selected_features], windowed_data['Cycle'], selected_features, feature_descriptions)
                    st.plotly_chart(fig_input, use_container_width=True, key=f"input_{cycle_num}")
                else:
                    st.write("No historical data to display.")

            # Update Output Visualization
            with output_placeholder.container():
                if config.MODEL_TYPE == 'binary':
                    # Binary Classification: Use Gauge
                    if not np.isnan(y_pred_prob):
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=y_pred_prob * 100,  # Convert to percentage
                            title={'text': "Probability (%)"},
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
                        fig_gauge.update_layout(width=600, height=400)
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_binary_{cycle_num}")
                    else:
                        st.write("Prediction Error")

                elif config.MODEL_TYPE == 'multiclass':
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

                elif config.MODEL_TYPE == 'regression':
                    # Regression: Display predicted value
                    st.write(f"Predicted Value at Cycle {cycle_num}: {y_pred_value:.2f}")

            # Simulate real-time by adding a delay
            time.sleep(speed)

        # Create a DataFrame for All Predictions
        if config.MODEL_TYPE == 'binary':
            pred_df = pd.DataFrame({
                'Cycle': cycle_list,
                'Predicted_Probability': y_pred_list
            })
        elif config.MODEL_TYPE == 'multiclass':
            # Convert list of numpy arrays to separate columns
            pred_df = pd.DataFrame(cycle_list, columns=['Cycle'])
            for cls_idx, cls in enumerate(classes):
                pred_df[f'Probability_{cls}'] = [prob[cls_idx] * 100 for prob in y_pred_list]
        elif config.MODEL_TYPE == 'regression':
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
        if config.MODEL_TYPE == 'binary':
            fig_pred = px.line(
                pred_df,
                x='Cycle',
                y='Predicted_Probability',
                title=f"Predicted Probability Over Cycles for Engine {selected_id}"
            )
            fig_pred.add_hline(y=config.BINARY_THRESHOLD * 100, line_dash='dash', line_color='red',
                               annotation_text=f'Threshold ({config.BINARY_THRESHOLD * 100}%)')
            fig_pred.update_yaxes(title_text='Predicted Probability (%)')
            fig_pred.update_xaxes(title_text='Cycle')
            st.plotly_chart(fig_pred, use_container_width=True, key=f"pred_binary_{selected_id}")

        elif config.MODEL_TYPE == 'multiclass':
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

        elif config.MODEL_TYPE == 'regression':
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


def run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols,
                                    nb_features):
    """
    Handle the Multiple Engines Simulation.

    Parameters:
    - config: Config object containing model and simulation parameters.
    - test_df: DataFrame containing test data.
    - engine_motor_mapping: Dictionary mapping engine IDs to motor IDs.
    - motors_df: DataFrame containing motor specifications.
    - sensors_df: DataFrame containing sensor metadata.
    - sequence_cols: List of sensor feature column names.
    - nb_features: Number of features in the input sequences.
    """
    st.subheader("Multiple Engines Simulation")
    st.write("Simulate streaming data and predictions for multiple engines simultaneously.")

    # Allow user to select number of engines
    num_engines = st.number_input("Select Number of Engines to Simulate", min_value=1,
                                  max_value=min(10, len(test_df['id'].unique())), value=5, step=1)

    # Option for random selection
    random_selection = st.checkbox("Randomly Select Engines", value=True)
    if random_selection:
        selected_ids = random.sample(list(test_df['id'].unique()), num_engines)
    else:
        selected_ids = st.multiselect("Select Engine IDs", options=test_df['id'].unique(),
                                      default=test_df['id'].unique()[:num_engines])

    if not selected_ids:
        st.warning("Please select at least one engine to simulate.")
        return

    # Feature selection
    all_features = ['voltage_input', 'current_limit', 'speed_control'] + list(sensors_df['Sensor Name'])
    feature_descriptions = {feat: feat.replace('_', ' ').title() for feat in
                            ['voltage_input', 'current_limit', 'speed_control']}
    feature_descriptions.update(
        {row['Sensor Name']: f"{row['Sensor Name']} - {row['Description']}" for _, row in sensors_df.iterrows()})

    selected_features = st.multiselect("Select Features to Visualize for All Engines", options=all_features,
                                       default=['voltage_input', 's1', 's2'])
    if not selected_features:
        st.warning("Please select at least one feature to visualize.")
        return

    # Simulation Speed Slider
    speed = st.slider("Select Simulation Speed (Seconds per Cycle)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Start/Stop Simulation
    start_button = st.button("Start Simulation")
    stop_button = st.button("Stop Simulation")

    if 'run_multiple_simulation' not in st.session_state:
        st.session_state['run_multiple_simulation'] = False
    if start_button:
        st.session_state['run_multiple_simulation'] = True
    if stop_button:
        st.session_state['run_multiple_simulation'] = False

    if st.session_state['run_multiple_simulation']:
        # Load the model
        try:
            pm_model = PredictiveMaintenanceModel(config, nb_features, config.MODEL_TYPE)
            pm_model.load_and_build_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Prepare data for selected engines
        engines_data = {}
        sequences_dict = {}
        max_cycles = 0
        engine_cycles_remaining = {}
        for engine_id in selected_ids:
            engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
            seq_gen = SequenceGenerator(engine_data, config.SEQUENCE_LENGTH, sequence_cols, config.OUTPUT_COLUMN)
            sequences, _ = seq_gen.generate_sequences()
            sequences_dict[engine_id] = sequences
            engines_data[engine_id] = engine_data
            num_cycles = len(sequences)
            max_cycles = max(max_cycles, num_cycles)
            engine_cycles_remaining[engine_id] = num_cycles  # Track remaining cycles per engine

        # Define classes for multiclass
        classes = config.class_labels if config.MODEL_TYPE == 'multiclass' and hasattr(config, 'class_labels') else [
            'Class_A', 'Class_B', 'Class_C']

        # Initialize DataFrames to store predictions
        pred_dfs = {engine_id: pd.DataFrame(
            columns=['Cycle', 'Predicted_Probability'] if config.MODEL_TYPE == 'binary' else (
                    ['Cycle'] + [f'Probability_{cls}' for cls in
                                 classes]) if config.MODEL_TYPE == 'multiclass' else ['Cycle', 'Predicted_Value'])
            for engine_id in selected_ids}

        # Initialize placeholders for each engine
        engine_placeholders = initialize_placeholders(selected_ids)

        # Simulation Loop
        cycle_idx = 0
        while st.session_state['run_multiple_simulation']:
            if all(cycles == 0 for cycles in engine_cycles_remaining.values()):
                st.write("All engines have completed their cycles.")
                break

            # Process each engine's prediction and update displays
            for engine_id in selected_ids:
                if engine_cycles_remaining[engine_id] > 0:
                    # Get the sequence for the current cycle
                    seq = sequences_dict[engine_id][cycle_idx]
                    seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)

                    # Initialize prediction variables
                    y_pred_prob, y_pred_probs, y_pred_value = None, None, None
                    try:
                        # Make prediction based on output type
                        if config.MODEL_TYPE == 'binary':
                            y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
                            y_pred_class = int(y_pred_prob > config.BINARY_THRESHOLD)
                        elif config.MODEL_TYPE == 'multiclass':
                            y_pred_probs = pm_model.model.predict(seq_reshaped)[0]
                            y_pred_class = np.argmax(y_pred_probs)
                            y_pred_prob = y_pred_probs[y_pred_class]
                        elif config.MODEL_TYPE == 'regression':
                            y_pred_value = pm_model.model.predict(seq_reshaped)[0][0]
                        else:
                            st.error(f"Unsupported OUTPUT_TYPE: {config.MODEL_TYPE}")
                            continue
                    except Exception as e:
                        st.error(f"Error during prediction for engine {engine_id} at cycle {cycle_idx + 1}: {e}")
                        continue

                    # Get cycle number
                    if cycle_idx < len(engines_data[engine_id]):
                        cycle_num = engines_data[engine_id]['cycle'].iloc[cycle_idx]
                    else:
                        cycle_num = f"Cycle_{cycle_idx + 1}"

                    # Accumulate predictions
                    if config.MODEL_TYPE == 'binary':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'Cycle': cycle_num,
                            'Predicted_Probability': y_pred_prob * 100  # Convert to percentage
                        }, ignore_index=True)
                    elif config.MODEL_TYPE == 'multiclass':
                        pred_entry = {'Cycle': cycle_num}
                        for cls_idx, cls in enumerate(classes):
                            pred_entry[f'Probability_{cls}'] = y_pred_probs[cls_idx] * 100  # Convert to percentage
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append(pred_entry, ignore_index=True)
                    elif config.MODEL_TYPE == 'regression':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'Cycle': cycle_num,
                            'Predicted_Value': y_pred_value
                        }, ignore_index=True)

                    # Update display using process_engine_prediction
                    process_engine_prediction(
                        engine_id=engine_id,
                        cycle_idx=cycle_idx,
                        engine_data=engines_data[engine_id],
                        placeholders=engine_placeholders[engine_id],
                        config=config,
                        selected_features=selected_features,
                        feature_descriptions=feature_descriptions,
                        pred_df=pred_dfs[engine_id],  # Pass the accumulated predictions
                        classes=classes
                    )

                    # Decrement the remaining cycles for the engine
                    engine_cycles_remaining[engine_id] -= 1
                else:
                    continue  # Skip to next engine

            # Control simulation speed
            time.sleep(speed)
            cycle_idx += 1

        # End of simulation
        st.session_state['run_multiple_simulation'] = False
        st.success("✅ Multiple Engines Simulation completed.")


def process_engine_prediction(engine_id, cycle_idx, engine_data, placeholders, config,
                              selected_features=None, feature_descriptions=None,
                              pred_df=None, classes=None):
    """
    Process and display predictions for a single engine at a given cycle index.
    """
    # Check if cycle_idx is within the engine data length
    if cycle_idx >= len(engine_data):
        return  # Skip if cycle index exceeds available data

    # Get the prediction for the current cycle from pred_df
    current_prediction = pred_df.iloc[-1]  # Assuming pred_df is updated up to current cycle
    cycle_num = current_prediction['Cycle']

    # Initialize variables for predicted values
    y_pred_prob, y_pred_probs, y_pred_value = None, None, None

    # Extract predicted values based on MODEL_TYPE
    if config.MODEL_TYPE == 'binary':
        y_pred_prob = current_prediction['Predicted_Probability'] / 100  # Convert back to [0,1]
    elif config.MODEL_TYPE == 'multiclass':
        prob_columns = [col for col in pred_df.columns if col.startswith('Probability_')]
        y_pred_probs = current_prediction[prob_columns].values / 100  # Convert back to [0,1]
    elif config.MODEL_TYPE == 'regression':
        y_pred_value = current_prediction['Predicted_Value']
    else:
        st.error(f"Unsupported OUTPUT_TYPE: {config.MODEL_TYPE}")
        return

    # Determine status indicator based on predictions
    status_indicator = ''
    if config.MODEL_TYPE == 'binary':
        threshold = config.BINARY_THRESHOLD * 100  # Convert to percentage
        status_indicator = config.STATUS_COLORS.get('safe',
                                                    '🟢') if y_pred_prob * 100 < threshold else config.STATUS_COLORS.get(
            'critical', '🔴')
    elif config.MODEL_TYPE == 'multiclass':
        y_pred_class = np.argmax(y_pred_probs)
        predicted_class_label = classes[y_pred_class] if classes else f"Class_{y_pred_class}"
        status_indicator = config.STATUS_COLORS.get(predicted_class_label, '⚪')  # Default to white if not found
    elif config.MODEL_TYPE == 'regression':
        threshold_value = config.REGRESSION_THRESHOLD
        status_indicator = config.STATUS_COLORS.get('safe',
                                                    '🟢') if y_pred_value < threshold_value else config.STATUS_COLORS.get(
            'critical', '🔴')

    # Write status indicator
    placeholders['status_placeholder'].write(f"{status_indicator} **Engine ID: {engine_id}**")

    # Use a unique key per engine per cycle
    unique_key = f"{engine_id}_{cycle_idx}"

    # Update Sensor Readings Plot if selected_features are provided
    if selected_features:
        historical_data = engine_data[selected_features].iloc[:cycle_idx + 1]
        cycles = engine_data['cycle'].iloc[:cycle_idx + 1]
        fig_sensor = plot_sensor_readings(historical_data, cycles, selected_features, feature_descriptions or {})
        placeholders['sensor_placeholder'].plotly_chart(fig_sensor, use_container_width=True,
                                                        key=f"sensor_{unique_key}")

    # Update Gauge or Display Based on Output Type
    if config.MODEL_TYPE == 'binary' and y_pred_prob is not None:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_pred_prob * 100,
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
        placeholders['gauge_placeholder'].plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{unique_key}")

    elif config.MODEL_TYPE == 'multiclass' and y_pred_probs is not None:
        class_labels = classes if classes else [f"Class_{i}" for i in range(len(y_pred_probs))]

        # Map class labels to colors
        colors = []
        default_color = 'gray'  # Default color if class label not in STATUS_COLORS
        for label in class_labels:
            color = config.LABEL_COLORS.get(label, default_color)
            colors.append(color)

        # Create the bar chart
        fig_bar = px.bar(
            x=class_labels,
            y=y_pred_probs * 100,
            labels={'x': 'Classes', 'y': 'Probability (%)'},
            title=f"Class Probabilities at Cycle {cycle_num}",
            text=y_pred_probs * 100
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color=colors)
        fig_bar.update_layout(yaxis=dict(range=[0, 100]), showlegend=False, height=250)
        placeholders['gauge_placeholder'].plotly_chart(fig_bar, use_container_width=True, key=f"bar_{unique_key}")

    elif config.MODEL_TYPE == 'regression' and y_pred_value is not None:
        placeholders['gauge_placeholder'].write(f"Predicted Value at Cycle {cycle_num}: {y_pred_value:.2f}",
                                                key=f"value_{unique_key}")

    # Update Prediction Graph using accumulated pred_df
    if pred_df is not None and not pred_df.empty:
        if config.MODEL_TYPE == 'binary':
            fig_pred = px.line(pred_df, x='Cycle', y='Predicted_Probability', title='', height=250)
            fig_pred.add_hline(y=config.BINARY_THRESHOLD * 100, line_dash='dash', line_color='red',
                               annotation_text=f'Threshold ({config.BINARY_THRESHOLD * 100}%)')
            fig_pred.update_yaxes(title_text='Predicted Probability (%)')
            fig_pred.update_xaxes(title_text='Cycle')
            placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                key=f"prediction_{unique_key}")

        elif config.MODEL_TYPE == 'multiclass':
            prob_columns = [col for col in pred_df.columns if col.startswith('Probability_')]
            fig_pred = px.line(pred_df, x='Cycle', y=prob_columns, title='', height=250)
            fig_pred.update_yaxes(title_text='Predicted Probability (%)')
            fig_pred.update_xaxes(title_text='Cycle')
            placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                key=f"prediction_{unique_key}")

        elif config.MODEL_TYPE == 'regression':
            fig_pred = px.line(pred_df, x='Cycle', y='Predicted_Value',
                               title=f"Predicted Values Over Cycles for Engine {engine_id}", height=250)
            fig_pred.update_yaxes(title_text='Predicted Value')
            fig_pred.update_xaxes(title_text='Cycle')
            placeholders['prediction_placeholder'].plotly_chart(fig_pred, use_container_width=True,
                                                                key=f"prediction_{unique_key}")
