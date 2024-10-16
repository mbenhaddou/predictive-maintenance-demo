import streamlit as st
st.set_page_config(layout='wide')
from predicted_maintenance import *

from helprs import *
sys.path.append('../')
from src.lstm.data_loader import DataLoader



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
    CONFIG_FILE = 'config.json'
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
            config.update_from_dict(config_data)

    # Ensure essential paths are set
    config.DATASET_PATH = '../Dataset/'
    config.OUTPUT_PATH = '../Output/'

    # Extract parameters from config
    dataset_path = config.DATASET_PATH
    sequence_length = config.SEQUENCE_LENGTH
    w1 = config.W1
    w0 = config.W0

    @st.cache_data()
    def load_data(_config, dataset_path, sequence_length, w1, w0):
        """Load data using DataLoader."""
        data_loader = DataLoader(_config)
        data_loader.read_data()
        data_loader.output_column = _config.OUTPUT_COLUMN
        return data_loader

    if options != "Introduction":
        data_loader = load_data(config, dataset_path, sequence_length, w1, w0)
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
