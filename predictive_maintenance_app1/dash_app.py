import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from utils.helpers import (
    plot_sensor_readings,
    plot_confusion_matrix,
    plot_roc_curve_binary,
    plot_multiclass_roc,
    plot_regression_predictions,
    initialize_placeholders
)
from model.predictive_maintenance_model import PredictiveMaintenanceModel
from model.sequence_generator import SequenceGenerator
import time

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Predictive Maintenance Dashboard"

# Placeholder for global variables (simulate session state)
simulation_states = {
    'run_simulation': False,
    'simulation_type': 'multiple',
    'speed': 0.5,
    'n_intervals': 0,
    'engine_data': {}
}

# Your configuration and data (replace with your actual data and config)
config = {
    'SEQUENCE_LENGTH': 50,
    'MODEL_TYPE': 'binary',  # 'binary', 'multiclass', or 'regression'
    'BINARY_THRESHOLD': 0.5,
    'OUTPUT_COLUMN': 'label',
    'class_labels': ['safe', 'warning', 'critical'],
    'STATUS_COLORS': {'safe': '🟢', 'warning': '🟡', 'critical': '🔴'},
    'LABEL_COLORS': {'safe': 'green', 'warning': 'yellow', 'critical': 'red'},
    'REGRESSION_THRESHOLD': 0.7
}

# Sample test DataFrame (replace with your actual test data)
test_df = pd.DataFrame({
    'id': np.random.choice(range(1, 21), size=1000),
    'cycle': np.tile(range(1, 51), 20),
    's1': np.random.rand(1000),
    's2': np.random.rand(1000),
    'label': np.random.randint(0, 2, 1000)
})

sequence_cols = ['s1', 's2']
nb_features = len(sequence_cols)

engine_motor_mapping = {engine_id: f"Motor_{engine_id}" for engine_id in test_df['id'].unique()}
motors_df = pd.DataFrame({
    'Manufacturer Part Number': [f"Motor_{i}" for i in range(1, 21)],
    'Brand': [f"Brand_{i}" for i in range(1, 21)],
    'Specification': [f"Spec_{i}" for i in range(1, 21)]
})
sensors_df = pd.DataFrame({
    'Sensor Name': ['s1', 's2'],
    'Description': ['Sensor 1 Description', 'Sensor 2 Description']
})

# App Layout
app.layout = dbc.Container([
    html.H1("Prediction Simulation"),
    html.P("Simulate streaming data and observe real-time predictions from the trained LSTM model."),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Simulation Type"),
            dcc.RadioItems(
                id='simulation-type',
                options=[
                    {'label': 'Single Engine Simulation', 'value': 'single'},
                    {'label': 'Multiple Engines Simulation', 'value': 'multiple'},
                    {'label': 'Map View Simulation', 'value': 'map'}
                ],
                value='multiple',
                labelStyle={'display': 'block'}
            ),
            html.Br(),
            dbc.Label("Select Simulation Speed (Seconds per Cycle)"),
            dcc.Slider(
                id='simulation-speed',
                min=0.1,
                max=2.0,
                step=0.1,
                value=0.5,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            dbc.Button("Start Simulation", id='start-button', color='success'),
            dbc.Button("Stop Simulation", id='stop-button', color='danger', style={'marginLeft': '10px'}),
            html.Br(),
            html.Div(id='simulation-controls')
        ], width=3),
        dbc.Col([
            html.Div(id='simulation-content')
        ], width=9)
    ]),
    dcc.Interval(id='simulation-interval', interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id='simulation-data')
], fluid=True)

# Global variable to store simulation state
simulation_state = {
    'running': False,
    'simulation_type': 'multiple',
    'speed': 0.5,
    'n_intervals': 0,
    'engine_data': {}
}

# Callback to handle simulation start/stop
@app.callback(
    Output('simulation-interval', 'disabled'),
    Output('simulation-data', 'data'),
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('simulation-type', 'value'),
     State('simulation-speed', 'value'),
     State('simulation-interval', 'disabled'),
     State('simulation-data', 'data')],
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks, simulation_type, speed, interval_disabled, sim_data):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'start-button' in changed_id:
        interval_disabled = False
        sim_data = sim_data or {}
        sim_data.update({
            'running': True,
            'simulation_type': simulation_type,
            'speed': speed,
            'n_intervals': 0
        })
    elif 'stop-button' in changed_id:
        interval_disabled = True
        sim_data = sim_data or {}
        sim_data['running'] = False
    return interval_disabled, sim_data

# Callback to render the simulation content based on the selected type
@app.callback(
    Output('simulation-content', 'children'),
    [Input('simulation-type', 'value')]
)
def render_simulation_content(simulation_type):
    if simulation_type == 'map':
        return html.Div([
            html.H4("Map View Simulation"),
            dcc.Graph(id='map-view-graph'),
            html.Div(id='cycle-info'),
            dcc.Store(id='map-simulation-store')
        ])
    elif simulation_type == 'single':
        engine_ids = test_df['id'].unique()
        return html.Div([
            html.H4("Single Engine Simulation"),
            dcc.Dropdown(
                id='engine-id-dropdown',
                options=[{'label': f'Engine {id}', 'value': id} for id in engine_ids],
                value=engine_ids[0]
            ),
            dbc.Label("Select Features to Visualize"),
            dcc.Checklist(
                id='feature-checklist',
                options=[{'label': col, 'value': col} for col in sequence_cols],
                value=sequence_cols[:2],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='single-engine-graph'),
            html.Div(id='single-engine-output'),
            dcc.Store(id='single-simulation-store')
        ])
    elif simulation_type == 'multiple':
        engine_ids = test_df['id'].unique()
        return html.Div([
            html.H4("Multiple Engines Simulation"),
            dbc.Label("Number of Engines to Simulate"),
            dcc.Input(id='num-engines-input', type='number', value=5, min=1, max=len(engine_ids)),
            html.Br(),
            dbc.Label("Select Features to Visualize"),
            dcc.Checklist(
                id='feature-checklist-multiple',
                options=[{'label': col, 'value': col} for col in sequence_cols],
                value=sequence_cols[:2],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='multiple-engines-graph'),
            html.Div(id='multiple-engines-output'),
            dcc.Store(id='multiple-simulation-store')
        ])

# Map View Simulation Callback
@app.callback(
    Output('map-view-graph', 'figure'),
    Output('cycle-info', 'children'),
    Input('simulation-interval', 'n_intervals'),
    State('simulation-data', 'data'),
    State('map-simulation-store', 'data')
)
def update_map_view(n_intervals, sim_data, map_store):
    if not sim_data or sim_data.get('simulation_type') != 'map' or not sim_data.get('running'):
        raise dash.exceptions.PreventUpdate

    # Initialize or update map simulation data
    if map_store is None or n_intervals == 0:
        engine_ids = test_df['id'].unique()
        num_engines = len(engine_ids)
        grid_size = int(np.ceil(np.sqrt(num_engines)))
        x_coords, y_coords = np.meshgrid(range(grid_size), range(grid_size))
        grid_data = pd.DataFrame({'x': x_coords.flatten()[:num_engines], 'y': y_coords.flatten()[:num_engines], 'id': engine_ids})
        map_store = grid_data.to_dict('records')
    else:
        grid_data = pd.DataFrame(map_store)

    # Simulate statuses
    statuses = ['safe', 'warning', 'critical']
    grid_data['status'] = np.random.choice(statuses, len(grid_data))
    grid_data['color'] = grid_data['status'].map({
        'safe': 'green',
        'warning': 'yellow',
        'critical': 'red'
    })

    # Create figure
    fig = px.scatter(
        grid_data, x='x', y='y',
        color='color',
        color_discrete_map={'green': 'green', 'yellow': 'yellow', 'red': 'red'},
        title=f'Cycle {n_intervals}: Engine Status Map'
    )
    fig.update_traces(marker=dict(size=15))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))

    cycle_info = f"Cycle: {n_intervals}"

    return fig, cycle_info

# Single Engine Simulation Callback
@app.callback(
    Output('single-engine-graph', 'figure'),
    Output('single-engine-output', 'children'),
    Input('simulation-interval', 'n_intervals'),
    State('simulation-data', 'data'),
    State('engine-id-dropdown', 'value'),
    State('feature-checklist', 'value'),
    State('single-simulation-store', 'data')
)
def update_single_engine(n_intervals, sim_data, engine_id, selected_features, single_store):
    if not sim_data or sim_data.get('simulation_type') != 'single' or not sim_data.get('running'):
        raise dash.exceptions.PreventUpdate

    # Initialize or update single simulation data
    if single_store is None or n_intervals == 0:
        engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
        seq_gen = SequenceGenerator(
            df=engine_data,
            sequence_length=config['SEQUENCE_LENGTH'],
            sequence_cols=sequence_cols,
            output_column=config['OUTPUT_COLUMN']
        )
        sequences, labels = seq_gen.generate_sequences()
        total_sequences = len(sequences)
        single_store = {
            'engine_data': engine_data.to_dict('records'),
            'sequences': [seq.tolist() for seq in sequences],
            'labels': labels.tolist(),
            'cycle_list': [],
            'y_pred_list': []
        }

    # Retrieve data from store
    engine_data = pd.DataFrame(single_store['engine_data'])
    sequences = [np.array(seq) for seq in single_store['sequences']]
    labels = np.array(single_store['labels'])
    cycle_list = single_store['cycle_list']
    y_pred_list = single_store['y_pred_list']

    # Current cycle index
    i = n_intervals % len(sequences)

    # Load the model (you may want to load it once and store globally)
    pm_model = PredictiveMaintenanceModel(config, nb_features, config['MODEL_TYPE'])
    pm_model.load_and_build_model()

    # Reshape sequence and make prediction
    seq = sequences[i]
    seq_reshaped = seq.reshape(1, config['SEQUENCE_LENGTH'], nb_features)
    if config['MODEL_TYPE'] == 'binary':
        y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
        y_pred_list.append(y_pred_prob)
    # Add logic for other output types if necessary

    # Update cycle list
    cycle_num = engine_data['cycle'].iloc[i] if i < len(engine_data) else f"Cycle_{i + 1}"
    cycle_list.append(cycle_num)

    # Update historical data for plotting
    historical_data = engine_data.iloc[:i+1]

    # Plot selected features
    fig = plot_sensor_readings(
        historical_data[selected_features],
        historical_data['cycle'],
        selected_features,
        {feat: feat for feat in selected_features}
    )

    # Update store
    single_store['cycle_list'] = cycle_list
    single_store['y_pred_list'] = y_pred_list

    # Simulate prediction output
    prediction_output = f"Predicted probability at cycle {cycle_num}: {y_pred_prob:.2f}"

    return fig, prediction_output

# Multiple Engines Simulation Callback
@app.callback(
    Output('multiple-engines-graph', 'figure'),
    Output('multiple-engines-output', 'children'),
    Input('simulation-interval', 'n_intervals'),
    State('simulation-data', 'data'),
    State('num-engines-input', 'value'),
    State('feature-checklist-multiple', 'value'),
    State('multiple-simulation-store', 'data')
)
def update_multiple_engines(n_intervals, sim_data, num_engines, selected_features, multiple_store):
    if not sim_data or sim_data.get('simulation_type') != 'multiple' or not sim_data.get('running'):
        raise dash.exceptions.PreventUpdate

    # Initialize or update multiple simulation data
    if multiple_store is None or n_intervals == 0:
        engine_ids = random.sample(list(test_df['id'].unique()), min(num_engines, len(test_df['id'].unique())))
        engines_data = {}
        sequences_dict = {}
        for engine_id in engine_ids:
            engine_data = test_df[test_df['id'] == engine_id].reset_index(drop=True)
            seq_gen = SequenceGenerator(engine_data, config['SEQUENCE_LENGTH'], sequence_cols, config['OUTPUT_COLUMN'])
            sequences, _ = seq_gen.generate_sequences()
            sequences_dict[engine_id] = [seq.tolist() for seq in sequences]
            engines_data[engine_id] = engine_data.to_dict('records')
        multiple_store = {
            'engine_ids': engine_ids,
            'engines_data': engines_data,
            'sequences_dict': sequences_dict,
            'engine_cycles': {engine_id: 0 for engine_id in engine_ids}
        }

    # Retrieve data from store
    engine_ids = multiple_store['engine_ids']
    engines_data = {eid: pd.DataFrame(data) for eid, data in multiple_store['engines_data'].items()}
    sequences_dict = {eid: [np.array(seq) for seq in seqs] for eid, seqs in multiple_store['sequences_dict'].items()}
    engine_cycles = multiple_store['engine_cycles']

    # Load the model (you may want to load it once and store globally)
    pm_model = PredictiveMaintenanceModel(config, nb_features, config['MODEL_TYPE'])
    pm_model.load_and_build_model()

    # Prepare data for plotting
    plot_data = []
    for engine_id in engine_ids:
        i = engine_cycles[engine_id]
        sequences = sequences_dict[engine_id]
        engine_data = engines_data[engine_id]

        if i >= len(sequences):
            continue

        # Make prediction
        seq = sequences[i]
        seq_reshaped = seq.reshape(1, config['SEQUENCE_LENGTH'], nb_features)
        if config['MODEL_TYPE'] == 'binary':
            y_pred_prob = pm_model.model.predict(seq_reshaped)[0][0]
            # You can store predictions if needed

        # Update engine cycle
        engine_cycles[engine_id] += 1

        # Prepare data for plotting
        data = engine_data.iloc[:i+1]
        for feat in selected_features:
            plot_data.append({
                'cycle': data['cycle'],
                'value': data[feat],
                'feature': feat,
                'engine_id': engine_id
            })

    # Update store
    multiple_store['engine_cycles'] = engine_cycles

    # Create plot
    if plot_data:
        plot_df = pd.concat([pd.DataFrame(d) for d in plot_data], ignore_index=True)
        fig = px.line(plot_df, x='cycle', y='value', color='engine_id', facet_col='feature', title='Multiple Engines Sensor Readings')
    else:
        fig = go.Figure()

    # Simulate prediction output
    prediction_output = f"Cycle {n_intervals}: Predictions updated for {len(engine_ids)} engines."

    return fig, prediction_output

if __name__ == '__main__':
    app.run_server(debug=True)
