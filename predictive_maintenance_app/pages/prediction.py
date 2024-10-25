# pages/prediction.py
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from model.model_registry import MODEL_REGISTRY  # Ensure correct import
from model.predictive_maintenance_model import PredictiveMaintenanceModel
from model.sequence_generator import SequenceGenerator
from utils.config import Config  # Base Config class
from utils.helpers import (
    plot_sensor_readings,
    initialize_placeholders
)

def display(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features):
    """Display the Prediction page with simulation options."""
    st.header("Prediction Simulation")
    st.write("""
        Simulate streaming data and observe real-time predictions from the trained model.
        Choose between simulating a single engine, multiple engines, or all engines on a map view.
    """)

    # Load the model once and pass it to simulation functions
    pm_model = load_model(config, nb_features)

    if pm_model is not None:
        run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols,
                                        nb_features, pm_model)
    else:
        st.error("Model could not be loaded. Please check the configuration and ensure the model is trained.")

def load_model(config: Config, nb_features: int) -> PredictiveMaintenanceModel:
    """
    Initialize and load the trained PredictiveMaintenanceModel.
    """
    try:
        # Retrieve the selected model name from the config
        selected_model_name = getattr(config, 'MODEL_NAME', None)

        if not selected_model_name:
            st.error("Model name not found in the configuration. Please select a model in the configuration page.")
            return None

        # Retrieve model information from the registry
        model_info = MODEL_REGISTRY.get(selected_model_name)
        if not model_info:
            st.error(f"Model '{selected_model_name}' not found in the MODEL_REGISTRY.")
            return None

        model_class = model_info.get("model_class")
        if not model_class:
            st.error(f"Model class for '{selected_model_name}' not defined.")
            return None

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
        st.success("✅ Trained model loaded successfully.")

        # Optionally, display model summary
        with st.expander("View Model Architecture"):
            model_summary = []
            pm_model.model.model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_str = "\n".join(model_summary)
            st.text(model_summary_str)

        return pm_model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_multiple_engines_simulation(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols,
                                    nb_features, pm_model):
    """
    Handle the Multiple Engines Simulation.
    """
    st.subheader("Multiple Engines Simulation")
    st.write("Simulate streaming data and predictions for multiple engines simultaneously.")

    # Initialize simulation state
    if 'run_multiple_simulation' not in st.session_state:
        st.session_state['run_multiple_simulation'] = False

    # Start/Stop Simulation Buttons
    start_button = st.button(
        "Start Simulation",
        key='start_multiple_simulation_unique'
    )
    stop_button = st.button(
        "Stop Simulation",
        key='stop_multiple_simulation_unique'
    )

    # Update simulation state
    if start_button:
        st.session_state['run_multiple_simulation'] = True
        # Reset health history when simulation starts
        st.session_state['health_history'] = {}
    if stop_button:
        st.session_state['run_multiple_simulation'] = False

    # Variable to control whether controls are disabled
    controls_disabled = st.session_state['run_multiple_simulation']

    # Wrap controls inside an expander
    with st.expander("Simulation Controls", expanded=not controls_disabled):
        # Allow user to select number of engines with unique key
        num_engines = st.number_input(
            "Select Number of Engines to Simulate",
            min_value=1,
            max_value=min(10, len(test_df['id'].unique())),
            value=5,
            step=1,
            key='num_engines_simulation_unique',
            disabled=controls_disabled
        )

        # Option for random selection with unique key
        random_selection = st.checkbox(
            "Randomly Select Engines",
            value=True,
            key='random_selection_unique',
            disabled=controls_disabled
        )
        if random_selection:
            selected_ids = random.sample(list(test_df['id'].unique()), num_engines)
        else:
            selected_ids = st.multiselect(
                "Select Engine IDs",
                options=test_df['id'].unique(),
                default=test_df['id'].unique()[:num_engines],
                key='selected_engine_ids_unique',
                disabled=controls_disabled
            )

        if not selected_ids:
            st.warning("Please select at least one engine to simulate.")
            return

        # Add radio button to select between 'Remaining Cycles' and 'Health Percentage'
        graph_mode = st.radio(
            "Select Graph Mode",
            options=['Remaining Cycles', 'Health Percentage (%)'],
            index=1,  # Default to 'Health Percentage'
            key='graph_mode_radio_unique',
            disabled=controls_disabled
        )

        # Add sliders for warning and critical thresholds
        st.write("Set Warning and Critical Thresholds:")
        threshold_warning = st.slider(
            "Warning Threshold (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            key='threshold_warning_slider_unique',
            disabled=controls_disabled
        )
        threshold_critical = st.slider(
            "Critical Threshold (%)",
            min_value=0,
            max_value=threshold_warning,
            value=15,
            step=1,
            key='threshold_critical_slider_unique',
            disabled=controls_disabled
        )

        # Feature selection with unique key
        all_features = ['voltage_input', 'current_limit', 'speed_control'] + list(sensors_df['Sensor Name'])
        feature_descriptions = {
            **{feat: feat.replace('_', ' ').title() for feat in ['voltage_input', 'current_limit', 'speed_control']},
            **{row['Sensor Name']: f"{row['Sensor Name']} - {row['Description']}" for _, row in sensors_df.iterrows()}
        }

        selected_features = st.multiselect(
            "Select Features to Visualize for All Engines",
            options=all_features,
            default=['voltage_input', 's2', 's3'],
            format_func=lambda x: feature_descriptions.get(x, x),
            help="Choose a subset of features to display in real-time.",
            key='multiple_engines_features_unique',
            disabled=controls_disabled
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
            key='multiple_engines_speed_slider_unique',
            disabled=controls_disabled
        )

    if st.session_state['run_multiple_simulation']:
        # Define classes for multiclass if applicable
        classes = config.class_labels if config.TASK_TYPE == 'multiclass' and hasattr(config, 'class_labels') else [
            'Class_A', 'Class_B', 'Class_C']
        max_value = 100
        # Prepare data for selected engines
        engines_data = {}
        sequences_dict = {}
        max_cycles = 0
        engine_cycles_remaining = {}
        for engine_id in selected_ids:
            engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
            seq_gen = SequenceGenerator(
                df=engine_data,
                sequence_length=config.SEQUENCE_LENGTH,
                sequence_cols=sequence_cols,
                output_column=config.OUTPUT_COLUMN
            )
            sequences, _ = seq_gen.generate_sequences()
            sequences_dict[engine_id] = sequences
            engine_cycles_remaining[engine_id] = len(sequences)
            engines_data[engine_id] = engine_data
            max_cycles = max(max_cycles, len(sequences))

        # Initialize DataFrames to store predictions
        pred_dfs = {engine_id: pd.DataFrame(
            columns=['cycle', 'Predicted_Probability'] if config.TASK_TYPE == 'binary' else (
                    ['cycle'] + [f'Probability_{cls}' for cls in classes]) if config.TASK_TYPE == 'multiclass' else ['cycle', 'Predicted_Value'])
            for engine_id in selected_ids}

        # Initialize placeholders for each engine
        engine_placeholders = initialize_placeholders(selected_ids)

        # Simulation Loop
        cycle_idx = 0
        while st.session_state['run_multiple_simulation']:
            # Check if all engines have completed their cycles
            if all(cycles == 0 for cycles in engine_cycles_remaining.values()):
                st.write("All engines have completed their cycles.")
                break

            # Process each engine's prediction and update displays
            for engine_id in selected_ids:
                if engine_cycles_remaining[engine_id] > 0:
                    # Get the sequence for the current cycle
                    seq = sequences_dict[engine_id][cycle_idx]
                    seq_reshaped = seq.reshape(1, config.SEQUENCE_LENGTH, nb_features)

                    # Make Prediction
                    try:
                        prediction = pm_model.predict(np.array(seq_reshaped).astype(np.float32))[0]
                        if config.TASK_TYPE == 'binary':
                            y_pred_prob = prediction[0]
                            y_pred_class = int(y_pred_prob > config.BINARY_THRESHOLD)
                        elif config.TASK_TYPE == 'multiclass':
                            y_pred_probs = prediction
                            y_pred_class = np.argmax(y_pred_probs)
                        elif config.TASK_TYPE == 'regression':
                            y_pred_value = prediction
                    except Exception as e:
                        st.error(f"Error during prediction for engine {engine_id} at cycle {cycle_idx + 1}: {e}")
                        continue  # Skip to next engine

                    # Determine the Corresponding Cycle Number
                    cycle_num = engines_data[engine_id]['cycle'].iloc[cycle_idx] if cycle_idx < len(engines_data[engine_id]) else f"Cycle_{cycle_idx + 1}"

                    # Append to Lists
                    if config.TASK_TYPE == 'binary':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'cycle': cycle_num,
                            'Predicted_Probability': y_pred_prob * 100  # Convert to percentage
                        }, ignore_index=True)
                    elif config.TASK_TYPE == 'multiclass':
                        pred_entry = {'cycle': cycle_num}
                        for cls_idx, cls in enumerate(classes):
                            pred_entry[f'Probability_{cls}'] = y_pred_probs[cls_idx] * 100  # Convert to percentage
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append(pred_entry, ignore_index=True)
                    elif config.TASK_TYPE == 'regression':
                        pred_dfs[engine_id] = pred_dfs[engine_id]._append({
                            'cycle': cycle_num,
                            'Predicted_Value': y_pred_value
                        }, ignore_index=True)
                    max_value=max(max_value, y_pred_value)
                    # Update display using process_engine_prediction
                    process_engine_prediction(
                        engine_id=engine_id,
                        cycle_idx=cycle_idx,
                        engine_data=engines_data[engine_id],
                        placeholders=engine_placeholders[engine_id],
                        config=config,
                        selected_features=selected_features,
                        feature_descriptions=feature_descriptions,
                        pred_df=pred_dfs[engine_id],
                        classes=classes,
                        graph_mode=graph_mode,
                        threshold_warning=threshold_warning,
                        threshold_critical=threshold_critical,
                        max_value=max_value
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
                              pred_df=None, classes=None, graph_mode=None, threshold_warning=30, threshold_critical=15, max_value=100 ):
    """
    Process and display predictions for a single engine at a given cycle index.

    Parameters:
    - engine_id (str/int): The unique identifier for the engine.
    - cycle_idx (int): The current cycle index.
    - engine_data (DataFrame): The DataFrame containing data for the engine.
    - placeholders (dict): A dictionary containing Streamlit placeholders for status, gauge, prediction, and history.
    - config (Config): The configuration object.
    - selected_features (list): List of selected features to visualize.
    - feature_descriptions (dict): Dictionary mapping feature names to descriptions.
    - pred_df (DataFrame): DataFrame containing accumulated predictions.
    - classes (list): List of class labels for multiclass classification.
    """
    # Ensure Plotly is imported within the function if not already
    import plotly.graph_objects as go

    # Check if cycle_idx is within the engine data length
    if cycle_idx >= len(engine_data):
        return  # Skip if cycle index exceeds available data

    # Get the prediction for the current cycle from pred_df
    current_prediction = pred_df.iloc[-1]  # Assuming pred_df is updated up to current cycle
    cycle_num = current_prediction['cycle']

    # Initialize variables for predicted values
    y_pred_prob, y_pred_probs, y_pred_value = None, None, None

    # Extract predicted values based on TASK_TYPE
    if config.TASK_TYPE == 'binary':
        y_pred_prob = current_prediction['Predicted_Probability'] / 100  # Convert back to [0,1]
    elif config.TASK_TYPE == 'multiclass':
        prob_columns = [f'Probability_{cls}' for cls in classes]
        y_pred_probs = current_prediction[prob_columns].values / 100  # Convert back to [0,1]
    elif config.TASK_TYPE == 'regression':
        y_pred_value = current_prediction['Predicted_Value']
    else:
        st.error(f"Unsupported TASK_TYPE: {config.TASK_TYPE}")
        return

    # Determine status indicator based on predictions
    status_indicator = ''

    # Write status indicator
    placeholders['status_placeholder'].markdown(f"{status_indicator} **Engine ID: {engine_id}**")

    # Update Sensor Readings Plot if selected_features are provided
    if selected_features:
        historical_data = engine_data[selected_features].iloc[:cycle_idx + 1]
        cycles = engine_data['cycle'].iloc[:cycle_idx + 1]
        fig_sensor = plot_sensor_readings(historical_data[selected_features], cycles, selected_features, feature_descriptions or {})
        placeholders['sensor_placeholder'].plotly_chart(fig_sensor, use_container_width=True, key=f"sensor_{engine_id}_{cycle_num}")

    # Handle Different TASK_TYPEs
    if config.TASK_TYPE == 'regression':
        # Regression: Health Indicator and History Curve
        if y_pred_value is not None:
            # Step 1: Define max_days dynamically
            max_days = pred_df['Predicted_Value'].max()
            if max_days is None or max_days <= 0:
                # Set a default value if max_days is invalid
                max_days = 60
                st.warning(f"Invalid MAX_DAYS detected for Engine {engine_id}. Using default value: {max_days} days.")
            else:
                st.session_state.setdefault('max_days', {})
                if engine_id not in st.session_state['max_days']:
                    st.session_state['max_days'][engine_id] = max_days
                else:
                    # Update max_days if the current prediction exceeds it
                    if y_pred_value > st.session_state['max_days'][engine_id]:
                        st.session_state['max_days'][engine_id] = y_pred_value
                max_days = st.session_state['max_days'][engine_id]

            # Step 2: Calculate plot_value
            if max_days > 0:
                health_percentage = (y_pred_value / max_days) * 100
                health_percentage = max(min(health_percentage, 100), 0)  # Clip between 0 and 100
            else:
                health_percentage = 0

            # Step 3: Determine thresholds and colors
            # Define thresholds
            #threshold_warning = 30  # Health percentage
            #threshold_critical = 15  # Health percentage

            # Determine color and status_text based on plot_value
            if health_percentage >=  threshold_warning:
                color = 'green'
                status_text = 'Healthy'
            elif threshold_critical <= health_percentage < threshold_warning:
                color = 'yellow'
                status_text = 'Warning'
            else:
                color = 'red'
                status_text = 'Critical'

            # Step 4: Plot Circular Health Indicator
            fig_health_indicator = create_circular_health_indicator(
                health_percentage=health_percentage,
                status_text=status_text,
                color=color,
                engine_id=engine_id,
                cycle_num=cycle_num
            )
            placeholders['gauge_placeholder'].plotly_chart(fig_health_indicator, use_container_width=True, key=f"health_indicator_{engine_id}_{cycle_num}")

            # Step 5: Plot Health Percentage History Curve
            plot_health_history_curve(
                engine_id=engine_id,
                cycle_num=cycle_num,
                plot_value=health_percentage if graph_mode == 'Health Percentage (%)' else y_pred_value,
                range_values=[0,max_value],
                placeholder=placeholders['prediction_placeholder'],
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical,
                graph_mode=graph_mode
            )

def create_circular_health_indicator(health_percentage, status_text, color, engine_id, cycle_num):
    """
    Creates a circular health indicator with an arc representing plot_value.

    Parameters:
    - plot_value (float): Health percentage (0-100).
    - status_text (str): Status text ('Healthy', 'Warning', 'Critical').
    - color (str): Color of the arc (e.g., theme's primary color).
    - engine_id (str/int): The ID of the engine.
    - cycle_num (int/str): Current cycle number.

    Returns:
    - fig (Plotly Figure): The Plotly figure object.
    """
    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st

    # Access theme colors from session state
    text_color = st.session_state.get('textColor', '#000000')  # Default to black if not set
    background_color = st.session_state.get('backgroundColor', '#FFFFFF')  # Default to white
    secondary_background_color = st.session_state.get('secondaryBackgroundColor', '#F0F2F6')  # Default

    # Calculate the angle for the arc (in radians)
    angle = (health_percentage / 100) * 360
    radians = np.deg2rad(np.linspace(0, angle, 100))
    x = 1 + 0.8 * np.cos(radians)  # 0.8 defines the radius
    y = 1 + 0.8 * np.sin(radians)

    # Create the figure
    fig = go.Figure()

    # Add the background circle outline
    fig.add_shape(
        type="circle",
        x0=0, y0=0,
        x1=2, y1=2,
        line=dict(color=secondary_background_color, width=5),
    )

    # Add the health arc
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color=color, width=5),
        name='Health',
        hoverinfo='skip',
        showlegend=False,
    ))

    # Add percentage annotation
    fig.add_annotation(
        text=f"{health_percentage:.1f}%",
        x=1,
        y=1.2,
        font=dict(size=20, color=text_color),
        showarrow=False,
        xref="x",
        yref="y",
        align='center',
        yanchor='middle',
        xanchor='center'
    )

    # Add status text annotation
    fig.add_annotation(
        text=f"{status_text}",
        x=1,
        y=0.8,
        font=dict(size=16, color=text_color),
        showarrow=False,
        xref="x",
        yref="y",
        align='center',
        yanchor='middle',
        xanchor='center'
    )

    # Set the layout to ensure the circle remains perfectly circular
    fig.update_layout(
        xaxis=dict(
            range=[0, 2],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[0, 2],
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
    )

    return fig
def plot_health_history_curve(engine_id, cycle_num, plot_value,range_values, placeholder, threshold_warning=30, threshold_critical=15, graph_mode='Health Percentage (%)'):
    """
    Plots the history of health percentages over cycles.

    Parameters:
    - engine_id (str/int): The ID of the engine.
    - cycle_num (int/str): Current cycle number.
    - plot_value (float): Current health percentage.
    - config (Config): Configuration object containing thresholds.
    - placeholder: Streamlit placeholder to update the plot.
    """
    import plotly.graph_objects as go

    # Initialize health history in session state if not already
    if 'health_history' not in st.session_state:
        st.session_state['health_history'] = {}
    if engine_id not in st.session_state['health_history']:
        st.session_state['health_history'][engine_id] = []

    # Append current plot_value to history
    st.session_state['health_history'][engine_id].append(plot_value)

    # Retrieve history
    history = st.session_state['health_history'][engine_id]
    cycles_history = list(range(1, len(history) + 1))



    # Create the figure
    fig = go.Figure()

    # Add the health trend line
    fig.add_trace(go.Scatter(
        x=cycles_history,
        y=history,
        mode='lines+markers',
        line=dict(color='blue'),
        name='Health Percentage'
    ))

    # Add horizontal lines for thresholds
    fig.add_hline(y=threshold_warning, line_dash='dash', line_color='orange',
                 annotation_text='Warning Threshold', annotation_position='bottom right')
    fig.add_hline(y=threshold_critical, line_dash='dash', line_color='red',
                 annotation_text='Critical Threshold', annotation_position='bottom right')

    # Update layout
    fig.update_layout(
        title=f"{graph_mode} Over Time - Engine {engine_id}",
        xaxis_title='Cycle',
        yaxis_title=f'{graph_mode}',
        yaxis=dict(range=range_values),
        showlegend=False,
        height=200,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Plot the figure in the placeholder
    placeholder.plotly_chart(fig, use_container_width=True, key=f"health_history_{engine_id}_{cycle_num}")


def plot_sensor_readings(historical_data, cycles, selected_features, feature_descriptions):
    """
    Plots selected sensor readings over cycles.

    Parameters:
    - historical_data (DataFrame): DataFrame containing historical sensor data.
    - cycles (Series): Series containing cycle numbers.
    - selected_features (list): List of selected features to visualize.
    - feature_descriptions (dict): Dictionary mapping feature names to descriptions.

    Returns:
    - fig (Plotly Figure): The Plotly figure object.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for feature in selected_features:
        if feature in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=cycles,
                y=historical_data[feature],
                mode='lines',
                name=feature,
                line=dict(width=2)
            ))

    fig.update_layout(
        title="Sensor Readings Over Cycles",
        xaxis_title='Cycle',
        yaxis_title='Sensor Values',
        height=200,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig
