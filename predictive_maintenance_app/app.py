# app.py

import os
import streamlit as st
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
import colorsys

import utils.theme
from utils.theme import ThemeColor

# Set page configuration
st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

# Preset theme colors
preset_colors: list[tuple[str, ThemeColor]] = [
    ("Default light", ThemeColor(
        primaryColor="#ff4b4b",
        backgroundColor="#ffffff",
        secondaryBackgroundColor="#f0f2f6",
        textColor="#31333F",
    )),
    ("Default dark", ThemeColor(
        primaryColor="#ff4b4b",
        backgroundColor="#0e1117",
        secondaryBackgroundColor="#262730",
        textColor="#fafafa",
    ))
]

theme_from_initial_config = utils.theme.get_config_theme_color()
if theme_from_initial_config:
    preset_colors.append(("From the config", theme_from_initial_config))

default_color = preset_colors[0][1]

def sync_rgb_to_hls(key: str):
    # HLS states are necessary for the HLS sliders.
    rgb = utils.theme.parse_hex(st.session_state[key])
    hls = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    st.session_state[f"{key}H"] = round(hls[0] * 360)
    st.session_state[f"{key}L"] = round(hls[1] * 100)
    st.session_state[f"{key}S"] = round(hls[2] * 100)

def set_color(key: str, color: str):
    st.session_state[key] = color
    sync_rgb_to_hls(key)

def reconcile_theme_config():
    keys = ['primaryColor', 'backgroundColor', 'secondaryBackgroundColor', 'textColor']
    has_changed = False
    for key in keys:
        if st._config.get_option(f'theme.{key}') != st.session_state[key]:
            st._config.set_option(f'theme.{key}', st.session_state[key])
            has_changed = True
    if has_changed:
        st.rerun()

def on_preset_color_selected():
    _, color = preset_colors[st.session_state.preset_color]
    set_color('primaryColor', color.primaryColor)
    set_color('backgroundColor', color.backgroundColor)
    set_color('secondaryBackgroundColor', color.secondaryBackgroundColor)
    set_color('textColor', color.textColor)
    reconcile_theme_config()

def main():
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

    # Move theme settings to the sidebar
    with st.sidebar:
        st.title("Theme Settings")

        st.selectbox(
            "Preset colors",
            key="preset_color",
            options=range(len(preset_colors)),
            format_func=lambda idx: preset_colors[idx][0],
            on_change=on_preset_color_selected,
            index=1
        )
        on_preset_color_selected()
        if st.button("🎨 Generate a random color scheme 🎲"):
            primary_color, text_color, basic_background, secondary_background = utils.theme.generate_color_scheme()
            set_color('primaryColor', primary_color)
            set_color('backgroundColor', basic_background)
            set_color('secondaryBackgroundColor', secondary_background)
            set_color('textColor', text_color)
            reconcile_theme_config()

        st.info("Select 'Custom Theme' in the settings dialog to see the effect")

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

    # Access current theme colors from session state
    primary_color = st.session_state.get('primaryColor', default_color.primaryColor)
    background_color = st.session_state.get('backgroundColor', default_color.backgroundColor)
    text_color = st.session_state.get('textColor', default_color.textColor)

    # Define menu styles dynamically based on the theme
    menu_styles = {
        "container": {"padding": "5!important", "background-color": background_color},
        "icon": {"color": primary_color, "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
            "color": text_color,
        },
        "nav-link-selected": {"background-color": primary_color},
    }

    # Create the navigation bar using option_menu
    selected = option_menu(
        menu_title=None,  # No title
        options=[
            "Introduction",
            "Data Exploration",
            "Model Configuration",
            "Model Training",
            "Model Evaluation",
            "Prediction"
        ],
        icons=[
            "house",
            "bar-chart",
            "gear",
            "play-circle",
            "clipboard-data",
            "graph-up-arrow"
        ],
        menu_icon="cast",  # Icon for the menu
        default_index=0,
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
