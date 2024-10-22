# app.py

import os

import streamlit as st
# Import the option_menu from streamlit_option_menu
from streamlit_option_menu import option_menu

# Import page modules
from pages import (
    introduction,
    data_exploration,
    model_configuration,
    model_training,
    model_evaluation,
    prediction
)
from utils.helpers import (
    load_csv,
    assign_motors_to_engines,
    load_or_initialize_config
)


def main():
    st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

    # Hide Streamlit's default elements (menu, footer, header, and sidebar toggle)
    hide_streamlit_style = """
                <style>
                /* Hide the hamburger menu */
                #MainMenu {visibility: hidden;}
                /* Hide the footer */
                footer {visibility: hidden;}
                /* Hide the header */
                header {visibility: hidden;}
                /* Hide the sidebar (expander) toggle button */
                [data-testid="collapsedControl"] {display: none;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Ready4 Model Prediction")
    st.write("""
        **Objective**: Predict if an engine will fail within a certain number of cycles using LSTM neural networks.
    """)

    # Load sensor metadata
    sensors_df = load_csv(os.path.join('Dataset', 'sensors.csv'))

    # Create mappings for sensor descriptions and abbreviations
    sensor_descriptions = dict(zip(sensors_df['Sensor Name'], sensors_df['Description']))
    sensor_abbreviations = dict(zip(sensors_df['Sensor Name'], sensors_df['Abbreviation']))

    # Load motor specifications
    motors_df = load_csv(os.path.join('Dataset', 'motor_specifications.csv'))

    config_dir = "configurations"
    os.makedirs(config_dir, exist_ok=True)
    config_file_path = os.path.join(config_dir, "config.json")

    # Load or initialize configuration
    config = load_or_initialize_config(config_file_path)


    # Ensure essential paths are set
    config.DATASET_PATH = os.path.join('Dataset', '')
    config.OUTPUT_PATH = os.path.join('../Output', '')

    # Extract parameters from config
    dataset_path = config.DATASET_PATH
    sequence_length = config.SEQUENCE_LENGTH
    w1 = config.W1
    w0 = config.W0

    # Load Data
    @st.cache_data()
    def load_data(_config, dataset_path, sequence_length, w1, w0):
        """Load data using DataLoader."""
        from model.data_loader import DataLoader  # Import here to avoid circular imports
        data_loader = DataLoader(_config)
        data_loader.read_data()
        data_loader.output_column = _config.OUTPUT_COLUMN
        return data_loader

    # Initialize variables only if not on the Introduction page
    menu_options = [
        "Introduction",
        "Data Exploration",
        "Model Configuration",
        "Model Training",
        "Model Evaluation",
        "Prediction"
    ]

    # Define icons for each menu item (optional)
    menu_icons = [
        "house",
        "bar-chart",
        "gear",
        "play-circle",
        "clipboard-data",
        "graph-up-arrow"
    ]

    # Define menu styles (optional)
    menu_styles = {
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"}
    }

    # Create the navigation bar using option_menu
    selected = option_menu(
        menu_title=None,  # No title
        options=menu_options,
        icons=menu_icons,
        menu_icon="cast",  # Icon for the menu
        default_index=5,
        orientation="horizontal",
        styles=menu_styles
    )

    # Load data and assign motors only if not on the Introduction page
    if selected != "Introduction":
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
    if selected == "Introduction":
        introduction.display()
    elif selected == "Data Exploration":
        data_exploration.display(config, train_df, test_df, motors_df, engine_motor_mapping, sensors_df)
    elif selected == "Model Configuration":
        model_configuration.display()
    elif selected == "Model Training":
        model_training.display(train_df, sequence_cols, nb_features)
    elif selected == "Model Evaluation":
        model_evaluation.display()
    elif selected == "Prediction":
        prediction.display(config, test_df, engine_motor_mapping, motors_df, sensors_df, sequence_cols, nb_features)
    else:
        st.write("Please select an option from the navigation bar.")


if __name__ == "__main__":
    main()
