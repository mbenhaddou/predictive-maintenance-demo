# pages/model_configuration.py

import streamlit as st
from utils.helpers import save_config

def display(config):
    """Display the Model Configuration page."""
    st.header("Model Configuration")
    st.write("Adjust model parameters below:")

    # ---------------------------
    # 1. Output Type Selection
    # ---------------------------
    output_type_options = ["binary", "multiclass", "regression"]

    # Determine the index based on current OUTPUT_TYPE
    if config.OUTPUT_TYPE in output_type_options:
        output_type_index = output_type_options.index(config.OUTPUT_TYPE)
    else:
        output_type_index = 0  # Default to 'binary' if not found

    selected_output_type = st.selectbox(
        "Select Output Type",
        output_type_options,
        index=output_type_index,
        key='output_type_selectbox'
    )
    config.OUTPUT_TYPE = selected_output_type  # Update the config instance

    # ---------------------------
    # 2. Set Output Column
    # ---------------------------
    if selected_output_type == "binary":
        config.OUTPUT_COLUMN = "label_binary"
    elif selected_output_type == "multiclass":
        config.OUTPUT_COLUMN = "label_multiclass"
    elif selected_output_type == "regression":
        config.OUTPUT_COLUMN = "RUL"

    # ---------------------------
    # 3. Model Parameters
    # ---------------------------

    # Dataset Path
    config.DATASET_PATH = st.text_input(
        "Dataset Path",
        value=config.DATASET_PATH,
        key='dataset_path_input'
    )

    # Sequence Length
    config.SEQUENCE_LENGTH = st.number_input(
        "Sequence Length",
        min_value=10,
        max_value=100,
        value=config.SEQUENCE_LENGTH,
        step=10,
        key='sequence_length_input'
    )

    # Threshold W1 for Label Generation
    config.W1 = st.number_input(
        "Threshold W1 for Label Generation",
        min_value=1,
        max_value=100,
        value=config.W1,
        step=5,
        key='w1_input'
    )

    # Threshold W0 for Label Generation
    config.W0 = st.number_input(
        "Threshold W0 for Label Generation",
        min_value=1,
        max_value=100,
        value=config.W0,
        step=5,
        key='w0_input'
    )

    # LSTM Units for Layer 1 and Layer 2
    config.LSTM_UNITS = [
        st.number_input(
            "LSTM Units Layer 1",
            min_value=32,
            max_value=256,
            value=config.LSTM_UNITS[0],
            step=32,
            key='lstm_units_layer1'
        ),
        st.number_input(
            "LSTM Units Layer 2",
            min_value=32,
            max_value=256,
            value=config.LSTM_UNITS[1],
            step=32,
            key='lstm_units_layer2'
        )
    ]

    # Dropout Rates for Layer 1 and Layer 2
    config.DROPOUT_RATES = [
        st.slider(
            "Dropout Rate Layer 1",
            min_value=0.0,
            max_value=0.5,
            value=config.DROPOUT_RATES[0],
            step=0.1,
            key='dropout_rate_layer1'
        ),
        st.slider(
            "Dropout Rate Layer 2",
            min_value=0.0,
            max_value=0.5,
            value=config.DROPOUT_RATES[1],
            step=0.1,
            key='dropout_rate_layer2'
        )
    ]

    # L2 Regularization
    config.L2_REG = st.number_input(
        "L2 Regularization",
        min_value=0.0,
        max_value=0.01,
        value=config.L2_REG,
        step=0.001,
        format="%.3f",
        key='l2_reg_input'
    )

    # Optimizer Selection
    optimizer_options = ["adam", "sgd"]
    if config.OPTIMIZER in optimizer_options:
        optimizer_index = optimizer_options.index(config.OPTIMIZER)
    else:
        optimizer_index = 0  # Default to 'adam' if not found

    config.OPTIMIZER = st.selectbox(
        "Optimizer",
        optimizer_options,
        index=optimizer_index,
        key='optimizer_selectbox'
    )

    # Learning Rate
    config.LEARNING_RATE = st.number_input(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=config.LEARNING_RATE,
        step=0.0001,
        format="%.4f",
        key='learning_rate_input'
    )

    # Epochs
    config.EPOCHS = st.number_input(
        "Epochs",
        min_value=10,
        max_value=200,
        value=config.EPOCHS,
        step=10,
        key='epochs_input'
    )

    # Batch Size
    config.BATCH_SIZE = st.number_input(
        "Batch Size",
        min_value=64,
        max_value=512,
        value=config.BATCH_SIZE,
        step=64,
        key='batch_size_input'
    )

    # ---------------------------
    # 4. Validation
    # ---------------------------
    if config.W1 <= config.W0:
        st.error("W1 should be greater than W0 for proper label generation.")

    # ---------------------------
    # 5. Save Configuration Button
    # ---------------------------
    if st.button("Save Configuration", key='save_config_button'):
        save_config(config)
