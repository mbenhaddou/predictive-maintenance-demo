# pages/model_configuration.py

import streamlit as st
import json
import os
from utils.helpers import get_models_by_task_type, apply_model_overrides
from model.model_registry import MODEL_REGISTRY
from utils.config import Config  # Base Config class
from model.lstm_model import LSTMConfig
from model.cnn_model import CNNConfig


def instantiate_config(config_class_name: str):
    """
    Instantiate the appropriate configuration class based on the class name.

    Args:
        config_class_name (str): The name of the configuration class.

    Returns:
        An instance of the specified configuration class.

    Raises:
        ValueError: If the configuration class name is unsupported.
    """
    if config_class_name == "LSTMConfig":
        return LSTMConfig()
    elif config_class_name == "CNNConfig":
        return CNNConfig()
    else:
        raise ValueError(f"Unsupported configuration class: {config_class_name}")


def display():
    """Display the Model Configuration page."""
    st.header("Model Configuration")
    st.write("Adjust model parameters below and save the configuration.")

    # ---------------------------
    # 1. Select Model
    # ---------------------------
    st.subheader("1. Select Model")
    model_names = list(MODEL_REGISTRY.keys())
    selected_model = st.selectbox("Choose a model:", model_names, key='model_selectbox')

    # Retrieve model information
    model_info = MODEL_REGISTRY.get(selected_model, {})
    config_class_name_full = model_info.get("config_class")  # Fully qualified class name

    if not config_class_name_full:
        st.error(f"No configuration class found for model '{selected_model}'.")
        return

    try:
        # Extract only the class name from the fully qualified class name
        if isinstance(config_class_name_full, str):
            config_class_name = config_class_name_full.split('.')[-1]
        else:
            config_class_name = config_class_name_full.__name__

        # Instantiate the appropriate configuration class
        config = instantiate_config(config_class_name)
    except ValueError as e:
        st.error(str(e))
        return

    # ---------------------------
    # 2. Display and Modify Configuration Parameters
    # ---------------------------
    st.subheader("2. Configure Parameters")

    # Iterate over dataclass fields and create input widgets
    for field_name, field_def in config.__dataclass_fields__.items():
        # Skip non-input fields or complex types
        if field_name in ["CALLBACKS_CONFIG", "class_labels", "STATUS_COLORS", "LABEL_COLORS"]:
            continue

        field_value = getattr(config, field_name)

        if isinstance(field_value, bool):
            # Boolean fields as checkboxes
            setattr(config, field_name, st.checkbox(
                f"{field_name.replace('_', ' ').title()}",
                value=field_value,
                key=field_name
            ))

        elif isinstance(field_value, int):
            # Integer fields as number inputs
            setattr(config, field_name, st.number_input(
                f"{field_name.replace('_', ' ').title()}",
                min_value=1,
                max_value=1000,
                value=field_value,
                step=1,
                key=field_name
            ))

        elif isinstance(field_value, float):
            # Float fields as number inputs with appropriate formatting
            if "lr" in field_name.lower() or "learning" in field_name.lower():
                step = 0.0001
                format_str = "%.4f"
            elif "reg" in field_name.lower():
                step = 0.001
                format_str = "%.3f"
            else:
                step = 0.1
                format_str = "%.2f"
            setattr(config, field_name, st.number_input(
                f"{field_name.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=100.0,
                value=field_value,
                step=step,
                format=format_str,
                key=field_name
            ))

        elif isinstance(field_value, list):
            # Handle lists of integers or floats with individual number inputs
            if all(isinstance(item, int) for item in field_value):
                new_list = []
                for idx, item in enumerate(field_value):
                    new_item = st.number_input(
                        f"{field_name.replace('_', ' ').title()} #{idx+1}",
                        min_value=1,
                        max_value=1000,
                        value=item,
                        step=1,
                        key=f"{field_name}_{idx}"
                    )
                    new_list.append(new_item)
                setattr(config, field_name, new_list)

            elif all(isinstance(item, float) for item in field_value):
                new_list = []
                for idx, item in enumerate(field_value):
                    new_item = st.number_input(
                        f"{field_name.replace('_', ' ').title()} #{idx+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=item,
                        step=0.05,
                        format="%.2f",
                        key=f"{field_name}_{idx}"
                    )
                    new_list.append(new_item)
                setattr(config, field_name, new_list)

        elif isinstance(field_value, str):
            # String fields as text inputs or selectboxes (e.g., optimizer)
            if field_name.lower() == "optimizer":
                optimizer_options = ["adam", "sgd"]
                optimizer_index = optimizer_options.index(field_value) if field_value in optimizer_options else 0
                setattr(config, field_name, st.selectbox(
                    f"{field_name.replace('_', ' ').title()}",
                    optimizer_options,
                    index=optimizer_index,
                    key=field_name
                ))
            else:
                setattr(config, field_name, st.text_input(
                    f"{field_name.replace('_', ' ').title()}",
                    value=field_value,
                    key=field_name
                ))

    # ---------------------------
    # 3. Validation
    # ---------------------------
    st.subheader("3. Validation")
    if hasattr(config, 'W1') and hasattr(config, 'W0'):
        if config.W1 <= config.W0:
            st.error("W1 should be greater than W0 for proper label generation.")
        else:
            st.success("Configuration is valid.")

    # ---------------------------
    # 4. Save Configuration Button
    # ---------------------------
    st.subheader("4. Save Configuration")
    save_config_button = st.button("Save Configuration", key='save_config_button')

    if save_config_button:
        try:
            # Prepare configuration dictionary
            config_dict = config.to_dict()
            config_dict["config_class"] = config_class_name  # Include only the class name
            config_dict["MODEL_NAME"] = selected_model      # Include the selected model name

            # Remove non-serializable fields
            non_serializable_fields = ["CALLBACKS_CONFIG", "class_labels", "STATUS_COLORS", "LABEL_COLORS"]
            for key in non_serializable_fields:
                if key in config_dict:
                    del config_dict[key]

            # Define the configuration file path
            # You can modify this path as needed
            config_dir = "../configurations"
            os.makedirs(config_dir, exist_ok=True)
            config_file_path = os.path.join(config_dir, "config.json")

            # Save the configuration to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

            st.success(f"Configuration saved successfully to `{config_file_path}`!")
            st.balloons()
        except TypeError as te:
            st.error(f"Failed to save configuration: {te}")
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
