# pages/model_configuration.py

import streamlit as st
import json
import os
from typing import Any, Dict, List
from utils.helpers import apply_model_overrides  # Ensure this function exists
from model.model_registry import MODEL_REGISTRY
from utils.config import Config  # Base Config class
from model.lstm_model import LSTMConfig
from model.cnn_model import CNNConfig

def apply_overrides(config, overrides):
    """
    Apply the overrides to the configuration instance.
    """
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            st.warning(f"Override key '{key}' is not a valid configuration attribute.")

def load_configuration(config_file_path: str):
    """
    Load the configuration from a JSON file and return a config object.
    """
    if not os.path.isfile(config_file_path):
        st.error(f"Configuration file `{config_file_path}` does not exist.")
        return None

    with open(config_file_path, 'r') as f:
        config_data = json.load(f)

    # Determine the config class from the saved data
    config_class_name = config_data.get("config_class")
    if not config_class_name:
        st.error("Config class name not found in the configuration file.")
        return None

    # Instantiate the appropriate configuration class
    if config_class_name == "LSTMConfig":
        config = LSTMConfig()
    elif config_class_name == "CNNConfig":
        config = CNNConfig()
    else:
        st.error(f"Unsupported configuration class: {config_class_name}")
        return None

    # Update the config object with the loaded data
    config.update_from_dict(config_data)
    st.success(f"Configuration loaded from `{config_file_path}`.")
    return config

def display():
    """Display the Model Configuration page."""
    st.header("Model Configuration")
    st.write("Adjust model parameters below and save or load configurations.")

    # ---------------------------
    # Configuration Directory
    # ---------------------------
    config_dir = "configurations"
    os.makedirs(config_dir, exist_ok=True)

    # ---------------------------
    # Load Configuration
    # ---------------------------
    st.subheader("Load Configuration")
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]

    if config_files:
        selected_config_file = st.selectbox("Select a configuration to load:", config_files, key='config_file_selectbox')
        load_button = st.button("Load Selected Configuration")
        if load_button:
            config_file_path = os.path.join(config_dir, selected_config_file)
            loaded_config = load_configuration(config_file_path)
            if loaded_config:
                st.session_state.config = loaded_config
                st.session_state.config_file_name = selected_config_file  # Store the config file name
                st.success(f"Configuration `{selected_config_file}` loaded successfully.")
                st.rerun()
    else:
        st.info("No configurations found. Please create and save a new configuration.")

    # ---------------------------
    # 1. Select Model
    # ---------------------------
    st.subheader("1. Select Model")
    model_names = list(MODEL_REGISTRY.keys())

    # Determine the initial selected model
    if st.session_state.get('config') and hasattr(st.session_state.config, 'MODEL_NAME'):
        selected_model = st.session_state.config.MODEL_NAME
        if selected_model not in model_names:
            st.warning(f"Model '{selected_model}' not found in the registry. Defaulting to the first model.")
            selected_model = model_names[0]
    else:
        selected_model = model_names[0]  # Default to the first model

    # Always render the selectbox
    selected_model = st.selectbox(
        "Choose a model:",
        model_names,
        index=model_names.index(selected_model),
        key='model_selectbox'
    )

    # Retrieve model information
    model_info = MODEL_REGISTRY.get(selected_model, {})
    config_class_ref = model_info.get("config_class")
    config_overrides = model_info.get("config_overrides", {})

    if not config_class_ref:
        st.error(f"No configuration class found for model '{selected_model}'.")
        return

    # Instantiate or update the configuration object
    try:
        if st.session_state.get('config') and st.session_state.config.MODEL_NAME == selected_model:
            # Use the loaded configuration
            config = st.session_state.config
        else:
            # Instantiate the appropriate configuration class
            config = config_class_ref()
            # Apply the overrides from the model registry
            apply_overrides(config, config_overrides)
            config.MODEL_NAME = selected_model  # Ensure the MODEL_NAME is set
            st.session_state.config = config  # Update session state

        # Keep track of parameters from the model registry
        registry_params = set(config_overrides.keys())
    except ValueError as e:
        st.error(str(e))
        return

    # ---------------------------
    # 2. Display and Modify Configuration Parameters
    # ---------------------------
    st.subheader("2. Configure Parameters")

    with st.form(key='config_form'):
        # General Parameters
        with st.expander("General Configuration"):
            general_fields = [
                'TASK_TYPE', 'CLASSIFICATION_TYPE', 'OUTPUT_COLUMN',
                'SEQUENCE_LENGTH', 'VALIDATION_SPLIT', 'RANDOM_STATE',
                'W1', 'W0'
            ]
            for field_name in general_fields:
                if not hasattr(config, field_name):
                    continue
                field_value = getattr(config, field_name)
                # Determine if the field should be disabled
                is_disabled = field_name in registry_params
                # Render the widget with the disabled parameter
                if field_name == 'TASK_TYPE':
                    options = config.TASK_TYPE_OPTIONS
                    index = options.index(field_value) if field_value in options else 0
                    config.TASK_TYPE = st.selectbox(
                        f"{field_name.replace('_', ' ').title()}",
                        options,
                        index=index,
                        key=field_name,
                        disabled=is_disabled
                    )
                elif field_name == 'CLASSIFICATION_TYPE':
                    if config.TASK_TYPE == 'classification':
                        options = config.CLASSIFICATION_TYPE_OPTIONS
                        index = options.index(field_value) if field_value in options else 0
                        config.CLASSIFICATION_TYPE = st.selectbox(
                            f"{field_name.replace('_', ' ').title()}",
                            options,
                            index=index,
                            key=field_name,
                            disabled=is_disabled
                        )
                elif isinstance(field_value, int):
                    setattr(config, field_name, st.number_input(
                        f"{field_name.replace('_', ' ').title()}",
                        min_value=1,
                        max_value=1000,
                        value=field_value,
                        step=1,
                        key=field_name,
                        disabled=is_disabled
                    ))
                elif isinstance(field_value, float):
                    setattr(config, field_name, st.number_input(
                        f"{field_name.replace('_', ' ').title()}",
                        min_value=0.0,
                        max_value=max(1.0, field_value * 5),
                        value=field_value,
                        step=0.05,
                        format="%.2f",
                        key=field_name,
                        disabled=is_disabled
                    ))
                elif isinstance(field_value, str):
                    setattr(config, field_name, st.text_input(
                        f"{field_name.replace('_', ' ').title()}",
                        value=field_value,
                        key=field_name,
                        disabled=is_disabled
                    ))

        # Model-Specific Parameters
        with st.expander("Model-Specific Configuration"):
            model_specific_fields = set(config.__dataclass_fields__.keys()) - set(general_fields)
            for field_name in model_specific_fields:
                # Skip non-input fields or complex types
                if field_name in ["CALLBACKS_CONFIG", "class_labels", "STATUS_COLORS", "LABEL_COLORS"]:
                    continue
                field_value = getattr(config, field_name)
                # Determine if the field should be disabled
                is_disabled = field_name in registry_params
                # Render the widget with the disabled parameter
                if isinstance(field_value, bool):
                    setattr(config, field_name, st.checkbox(
                        f"{field_name.replace('_', ' ').title()}",
                        value=field_value,
                        key=field_name,
                        disabled=is_disabled
                    ))
                elif isinstance(field_value, int):
                    setattr(config, field_name, st.number_input(
                        f"{field_name.replace('_', ' ').title()}",
                        min_value=1,
                        max_value=1000,
                        value=field_value,
                        step=1,
                        key=field_name,
                        disabled=is_disabled
                    ))
                elif isinstance(field_value, float):
                    # Float fields with appropriate formatting
                    if "lr" in field_name.lower() or "learning" in field_name.lower():
                        step = 0.0001
                        format_str = "%.4f"
                    elif "reg" in field_name.lower():
                        step = 0.001
                        format_str = "%.3f"
                    else:
                        step = 0.05
                        format_str = "%.2f"
                    setattr(config, field_name, st.number_input(
                        f"{field_name.replace('_', ' ').title()}",
                        min_value=0.0,
                        max_value=100.0,
                        value=field_value,
                        step=step,
                        format=format_str,
                        key=field_name,
                        disabled=is_disabled
                    ))
                elif isinstance(field_value, list):
                    # Handle lists with individual inputs
                    # Apply 'disabled' to all related widgets
                    if all(isinstance(item, int) for item in field_value):
                        new_list = []
                        count = st.number_input(
                            f"Number of {field_name.replace('_', ' ').title()}",
                            min_value=1,
                            max_value=10,
                            value=len(field_value),
                            step=1,
                            key=f"{field_name}_count",
                            disabled=is_disabled
                        )
                        for idx in range(int(count)):
                            if idx < len(field_value):
                                default_value = field_value[idx]
                            else:
                                default_value = 0
                            new_item = st.number_input(
                                f"{field_name.replace('_', ' ').title()} #{idx + 1}",
                                min_value=0,
                                max_value=1000,
                                value=default_value,
                                step=1,
                                key=f"{field_name}_{idx}",
                                disabled=is_disabled
                            )
                            new_list.append(new_item)
                        setattr(config, field_name, new_list)
                    elif all(isinstance(item, float) for item in field_value):
                        new_list = []
                        count = st.number_input(
                            f"Number of {field_name.replace('_', ' ').title()}",
                            min_value=1,
                            max_value=10,
                            value=len(field_value),
                            step=1,
                            key=f"{field_name}_count",
                            disabled=is_disabled
                        )
                        for idx in range(int(count)):
                            if idx < len(field_value):
                                default_value = field_value[idx]
                            else:
                                default_value = 0.0
                            new_item = st.number_input(
                                f"{field_name.replace('_', ' ').title()} #{idx + 1}",
                                min_value=0.0,
                                max_value=max(1.0, default_value * 5),
                                value=default_value,
                                step=0.05,
                                format="%.2f",
                                key=f"{field_name}_{idx}",
                                disabled=is_disabled
                            )
                            new_list.append(new_item)
                        setattr(config, field_name, new_list)
                elif isinstance(field_value, str):
                    if field_name.lower() == "optimizer":
                        optimizer_options = ["adam", "sgd"]
                        optimizer_index = optimizer_options.index(
                            field_value) if field_value in optimizer_options else 0
                        setattr(config, field_name, st.selectbox(
                            f"{field_name.replace('_', ' ').title()}",
                            optimizer_options,
                            index=optimizer_index,
                            key=field_name,
                            disabled=is_disabled
                        ))
                    else:
                        setattr(config, field_name, st.text_input(
                            f"{field_name.replace('_', ' ').title()}",
                            value=field_value,
                            key=field_name,
                            disabled=is_disabled
                        ))

        # Submit Button
        submit_button = st.form_submit_button(label='Save Configuration')

    # ---------------------------
    # 3. Validation
    # ---------------------------
    st.subheader("3. Validation")

    errors = []

    # Example validation
    if hasattr(config, 'LSTM_UNITS') and hasattr(config, 'DROPOUT_RATES'):
        if len(config.LSTM_UNITS) != len(config.DROPOUT_RATES):
            errors.append("The number of LSTM units must match the number of dropout rates.")

    if hasattr(config, 'CNN_FILTERS') and hasattr(config, 'DROPOUT_RATES'):
        if len(config.CNN_FILTERS) != len(config.DROPOUT_RATES):
            errors.append("The number of CNN filters must match the number of dropout rates.")

    if hasattr(config, 'W1') and hasattr(config, 'W0'):
        if config.W1 <= config.W0:
            errors.append("W1 should be greater than W0 for proper label generation.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        st.success("Configuration is valid.")

    # ---------------------------
    # 4. Save Configuration
    # ---------------------------
    st.subheader("4. Save Configuration")

    # Ask user for configuration name
    if st.session_state.get('config_file_name'):
        default_config_name = st.session_state['config_file_name'].replace('.json', '')
    else:
        default_config_name = 'config'

    config_name = st.text_input(
        "Enter a name for this configuration (without extension):",
        value=default_config_name,
        key='config_name_input'
    )

    # Prepare the config file path
    config_file_name = f"{config_name}.json"
    config_file_path = os.path.join(config_dir, config_file_name)

    # Check if file exists
    file_exists = os.path.exists(config_file_path)

    # Display overwrite warning and checkbox if file exists
    if file_exists:
        st.warning(f"A configuration named `{config_file_name}` already exists.")
        overwrite_confirmed = st.checkbox("I want to overwrite the existing file", key='overwrite_confirm_checkbox')
    else:
        overwrite_confirmed = True  # No need to confirm overwrite if file doesn't exist

    # Disable the save button if the config_name is empty or contains only whitespace, or if overwrite not confirmed
    submit_button_disabled = not bool(config_name.strip()) or (file_exists and not overwrite_confirmed)

    # Save Configuration Button
    submit_button = st.button(
        label='Save Configuration',
        disabled=submit_button_disabled
    )

    if submit_button:
        if errors:
            st.error("Please fix the errors above before saving the configuration.")
        else:
            try:
                # Prepare configuration dictionary
                config_dict = config.to_dict()
                config_dict["config_class"] = config.__class__.__name__  # Include the class name
                config_dict["MODEL_NAME"] = selected_model  # Include the selected model name

                # Save the configuration to a JSON file
                with open(config_file_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)

                st.success(f"Configuration saved successfully as `{config_file_name}`!")
                st.balloons()

                # Update session state
                st.session_state.config = config
                st.session_state.config_file_name = config_file_name
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    # Update the session state with the current config
    st.session_state.config = config
