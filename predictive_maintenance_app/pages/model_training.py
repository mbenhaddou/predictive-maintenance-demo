# pages/model_training.py

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Union
from model.model_registry import MODEL_REGISTRY
from utils.helpers import instantiate_config
from utils.config import Config
from model.lstm_model import LSTMConfig
from model.cnn_model import CNNConfig
from model.callbacks import StreamlitCallback
from model.predictive_maintenance_model import PredictiveMaintenanceModel
import plotly.express as px
import json
import os
from datetime import datetime

def load_configuration(config_file_path: str) -> Union[LSTMConfig, CNNConfig]:
    """
    Load the configuration from a JSON file and instantiate the appropriate Config subclass.

    Args:
        config_file_path (str): Path to the configuration JSON file.

    Returns:
        Union[LSTMConfig, CNNConfig]: An instance of the appropriate Config subclass.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the config_class is unsupported or missing.
    """
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' does not exist.")

    with open(config_file_path, 'r') as f:
        config_data = json.load(f)

    config_class_name = config_data.get("config_class")
    model_name = config_data.get("MODEL_NAME")

    if not config_class_name:
        raise ValueError("The configuration file is missing the 'config_class' field.")

    # Instantiate the appropriate Config subclass
    if config_class_name == "LSTMConfig":
        config = LSTMConfig()
    elif config_class_name == "CNNConfig":
        config = CNNConfig()
    else:
        raise ValueError(f"Unsupported config_class '{config_class_name}' in configuration file.")

    # Remove keys that are not part of the Config subclass to avoid unexpected attributes
    ignored_keys = ["config_class", "MODEL_NAME"]
    for key, value in config_data.items():
        if key in ignored_keys:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            st.warning(f"Ignoring unknown configuration parameter: {key}")

    # Assign the MODEL_NAME to the config if needed
    config.MODEL_NAME = model_name

    return config

def get_saved_models(output_dir='models/'):
    """
    Retrieve a list of saved model file paths from the output directory.

    Args:
        output_dir (str): Directory where models are saved.

    Returns:
        List[str]: List of model file paths.
    """
    supported_extensions = ['.h5', '.keras', '.tf']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model_files = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1].lower() in supported_extensions]
    model_paths = [os.path.join(output_dir, f) for f in model_files]
    return model_paths

def initialize_session_state():
    """
    Initialize necessary session state variables.
    """
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'trained_history' not in st.session_state:
        st.session_state.trained_history = None

def display_model_summary(pm_model: PredictiveMaintenanceModel, section_name: str = "Model Summary"):
    """
    Display the model summary in an expandable section.

    Args:
        pm_model (PredictiveMaintenanceModel): The model instance.
        section_name (str): Name of the section.
    """
    st.subheader(section_name)
    with st.expander("View Model Architecture"):
        model_summary = []
        pm_model.model.model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_str = "\n".join(model_summary)
        st.text(model_summary_str)

def display_training_history(history):
    """
    Display training history plots based on the task type.

    Args:
        history: Training history object.
    """
    st.subheader("4. Training History")
    history_df = pd.DataFrame(history.history)

    # Display relevant plots based on the task type
    # Classification Metrics
    if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        fig_acc = px.line(
            history_df,
            y=['accuracy', 'val_accuracy'],
            labels={'index': 'Epoch', 'value': 'Accuracy', 'variable': 'Dataset'},
            title='Model Accuracy Over Epochs'
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    elif 'acc' in history_df.columns and 'val_acc' in history_df.columns:
        # Handle older Keras versions that use 'acc' instead of 'accuracy'
        fig_acc = px.line(
            history_df,
            y=['acc', 'val_acc'],
            labels={'index': 'Epoch', 'value': 'Accuracy', 'variable': 'Dataset'},
            title='Model Accuracy Over Epochs'
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.warning("⚠️ Accuracy metrics not available for the selected task type.")

    # Loss Plot
    if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
        fig_loss = px.line(
            history_df,
            y=['loss', 'val_loss'],
            labels={'index': 'Epoch', 'value': 'Loss', 'variable': 'Dataset'},
            title='Model Loss Over Epochs'
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.warning("⚠️ Loss metrics not available.")

    # Regression Metrics
    if 'mae' in history_df.columns and 'val_mae' in history_df.columns:
        fig_mae = px.line(
            history_df,
            y=['mae', 'val_mae'],
            labels={'index': 'Epoch', 'value': 'MAE', 'variable': 'Dataset'},
            title='Model MAE Over Epochs'
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    elif 'mean_absolute_error' in history_df.columns and 'val_mean_absolute_error' in history_df.columns:
        # Handle different naming
        fig_mae = px.line(
            history_df,
            y=['mean_absolute_error', 'val_mean_absolute_error'],
            labels={'index': 'Epoch', 'value': 'MAE', 'variable': 'Dataset'},
            title='Model MAE Over Epochs'
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.warning("⚠️ MAE metrics not available.")

    if 'mse' in history_df.columns and 'val_mse' in history_df.columns:
        fig_mse = px.line(
            history_df,
            y=['mse', 'val_mse'],
            labels={'index': 'Epoch', 'value': 'MSE', 'variable': 'Dataset'},
            title='Model MSE Over Epochs'
        )
        st.plotly_chart(fig_mse, use_container_width=True)
    elif 'mean_squared_error' in history_df.columns and 'val_mean_squared_error' in history_df.columns:
        fig_mse = px.line(
            history_df,
            y=['mean_squared_error', 'val_mean_squared_error'],
            labels={'index': 'Epoch', 'value': 'MSE', 'variable': 'Dataset'},
            title='Model MSE Over Epochs'
        )
        st.plotly_chart(fig_mse, use_container_width=True)
    else:
        st.warning("⚠️ MSE metrics not available.")

def display(train_df: pd.DataFrame, sequence_cols: List[str], nb_features: int):
    """Display the Model Training and Management page."""
    initialize_session_state()

    st.header("Model Training and Management")
    st.write("Train your model with the configured parameters below or load an existing model for evaluation.")

    # ---------------------------
    # 1. Load Configuration
    # ---------------------------
    st.subheader("1. Load Configuration")
    config_file_path = "configurations/config.json"  # Define the path to your configuration file

    try:
        config = load_configuration(config_file_path)
        st.success(f"Configuration loaded successfully from `{config_file_path}`!")
    except FileNotFoundError as fnf_error:
        st.error(str(fnf_error))
        st.stop()
    except ValueError as ve:
        st.error(str(ve))
        st.stop()
    except json.JSONDecodeError:
        st.error("The configuration file contains invalid JSON.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the configuration: {e}")
        st.stop()

    # Optionally, display the loaded configuration
    with st.expander("View Loaded Configuration"):
        config_dict = config.to_dict()
        config_dict["MODEL_NAME"] = config.MODEL_NAME  # Ensure MODEL_NAME is included
        st.json(config_dict)

    # ---------------------------
    # 2. Generate Sequences and Labels for Training
    # ---------------------------
    st.subheader("2. Prepare Training Data")
    try:
        from model.sequence_generator import SequenceGenerator

        seq_gen = SequenceGenerator(
            df=train_df,
            sequence_length=config.SEQUENCE_LENGTH,
            sequence_cols=sequence_cols,
            output_column=config.OUTPUT_COLUMN
        )
        sequences, labels = seq_gen.generate_sequences(padding_strategy='zero')
        st.success(f"Generated {len(sequences)} sequences for training.")
    except Exception as e:
        st.error(f"Error in generating sequences: {e}")
        st.stop()

    # ---------------------------
    # 3. Training Parameters and Training
    # ---------------------------
    st.subheader("3. Training Parameters")
    st.write("Review and adjust the training parameters as needed.")

    # Button to initiate training
    if st.button("Train Model", key='train_model_button_unique'):
        try:
            # Retrieve the selected model name from the config
            selected_model_name = getattr(config, 'MODEL_NAME', None)

            if not selected_model_name:
                st.error("Model name not found in the configuration. Please select a model in the configuration page.")
                st.stop()

            # Retrieve model information from the registry
            model_info = MODEL_REGISTRY.get(selected_model_name)
            if not model_info:
                st.error(f"Model '{selected_model_name}' not found in the MODEL_REGISTRY.")
                st.stop()

            model_class = model_info.get("model_class")
            if not model_class:
                st.error(f"Model class for '{selected_model_name}' not defined.")
                st.stop()

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

            # Build and compile the model with appropriate input shape
            pm_model.build_and_compile_model()
            st.success("Model built and compiled successfully.")

            # Display model summary
            display_model_summary(pm_model, section_name="Model Summary")

            # Placeholders for progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create the Streamlit callback with task_type
            st_callback = StreamlitCallback(
                epochs=config.EPOCHS,
                progress_bar=progress_bar,
                status_text=status_text,
                output_type=config.TASK_TYPE  # Updated to use TASK_TYPE
            )

            # Train the model with the custom callback
            history = pm_model.train(
                seq_array=np.array(sequences).astype(np.float32),
                label_array=np.array(labels).astype(np.float32),
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT,
                custom_callback=[st_callback]  # Pass the custom callback as a list
            )

            st.success("✅ Model training completed successfully!")

            # Store the trained model and history in session_state
            st.session_state.trained_model = pm_model
            st.session_state.trained_history = history.history

            # ---------------------------
            # 4. Plot Training History
            # ---------------------------
            display_training_history(history)

        except Exception as e:
            st.error(f"❌ An error occurred during training: {e}")

    # ---------------------------
    # 4. Save the Trained Model
    # ---------------------------
    st.subheader("4. Save Trained Model")
    if st.session_state.trained_model:
        if st.button("Save Model", key='save_model_button_unique'):
            try:
                # Define the save path, e.g., based on timestamp
                save_path = config.get_model_path()
                st.session_state.trained_model.save_full_model(filepath=save_path)
                st.success(f"✅ Model saved successfully at `{save_path}`!")
                # Clear the trained model from session_state
                st.session_state.trained_model = None
                st.session_state.trained_history = None
            except Exception as e:
                st.error(f"❌ Failed to save model: {e}")
    else:
        st.info("No trained model available to save. Train a model first.")

    # ---------------------------
    # 5. Load and Verify Existing Models
    # ---------------------------
    st.subheader("5. Load and Verify Existing Models")
    output_dir = config.OUTPUT_PATH  # Define your models directory
    saved_models = get_saved_models(output_dir=output_dir)

    if saved_models:
        st.info(f"Found {len(saved_models)} model(s) in `{output_dir}`.")
        selected_model = st.selectbox("Select a model to load and verify:", saved_models)

        if st.button("Load and Verify Model", key='load_verify_model_button_unique'):
            try:
                # Retrieve the selected model name from the config
                selected_model_name = getattr(config, 'MODEL_NAME', None)

                if not selected_model_name:
                    st.error("Model name not found in the configuration. Please check your configuration file.")
                    st.stop()

                # Retrieve model information from the registry
                model_info = MODEL_REGISTRY.get(selected_model_name)
                if not model_info:
                    st.error(f"Model '{selected_model_name}' not found in the MODEL_REGISTRY.")
                    st.stop()

                model_class = model_info.get("model_class")
                if not model_class:
                    st.error(f"Model class for '{selected_model_name}' not defined.")
                    st.stop()

                # Instantiate the specific model class
                loaded_model_instance = model_class(
                    config=config,
                    nb_features=nb_features
                )

                # Wrap the model instance with PredictiveMaintenanceModel
                loaded_pm_model = PredictiveMaintenanceModel(
                    model=loaded_model_instance,
                    nb_features=nb_features,
                    config=config
                )

                # Load the selected model
                loaded_pm_model.load_full_model(load_path=selected_model)
                st.success(f"✅ Model loaded successfully from `{selected_model}`!")

                # Make predictions with the loaded model to verify
                sample_input = sequences[:10] if len(sequences) >=10 else sequences
                sample_input = np.array(sample_input).astype(np.float32)
                predictions = loaded_pm_model.predict(sample_input)
                st.write(f"**Predictions on Sample Data:**\n{predictions}")

                # Display model summary
                display_model_summary(loaded_pm_model, section_name="Loaded Model Summary")

            except Exception as e:
                st.error(f"❌ Failed to load or verify model: {e}")
    else:
        st.warning(f"No models found in `{output_dir}`. Please train and save a model first.")
