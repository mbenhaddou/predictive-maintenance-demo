# pages/model_training.py

import streamlit as st
import numpy as np
import pandas as pd
from model.predictive_maintenance import SequenceGenerator, PredictiveMaintenanceModel
from model.callbacks import StreamlitCallback
import plotly.express as px

def display(config, train_df, sequence_cols, nb_features):
    """Display the Model Training page."""
    st.header("Model Training")
    st.write("Train your LSTM model with the configured parameters below.")

    # Generate sequences and labels for training
    try:
        seq_gen = SequenceGenerator(
            df=train_df,
            sequence_length=config.SEQUENCE_LENGTH,
            sequence_cols=sequence_cols,
            output_column=config.OUTPUT_COLUMN
        )
        sequences, labels = seq_gen.generate_sequences()
        st.success(f"Generated {len(sequences)} sequences for training.")
    except Exception as e:
        st.error(f"Error in generating sequences: {e}")
        return

    # Training Parameters
    st.subheader("Training Parameters")
    st.write("Review and adjust the training parameters as needed.")

    # Button to initiate training
    if st.button("Train Model", key='train_model_button'):
        try:
            # Initialize PredictiveMaintenanceModel
            pm_model = PredictiveMaintenanceModel(
                config=config,
                nb_features=nb_features,
                output_type=config.OUTPUT_TYPE
            )
            pm_model.build_model()

            # Placeholders for progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create the Streamlit callback with output_type
            st_callback = StreamlitCallback(
                epochs=config.EPOCHS,
                progress_bar=progress_bar,
                status_text=status_text,
                output_type=config.OUTPUT_TYPE
            )

            # Train model with the custom callback
            history = pm_model.train_model(
                seq_array=np.array(sequences),
                label_array=np.array(labels),
                epochs=config.EPOCHS,
                validation_split=config.VALIDATION_SPLIT,
                batch_size=config.BATCH_SIZE,
                custom_callback=[st_callback]  # Pass the custom callback as a list
            )

            st.success("✅ Model training completed successfully!")

            # Plot training history based on output_type
            st.subheader("Training History")
            history_df = pd.DataFrame(history.history)

            # Display relevant plots based on the output type
            if config.OUTPUT_TYPE in ['binary', 'multiclass']:
                # Accuracy Plot
                if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
                    fig_acc = px.line(
                        history_df,
                        y=['accuracy', 'val_accuracy'],
                        labels={'index': 'Epoch', 'value': 'Accuracy', 'variable': 'Dataset'},
                        title='Model Accuracy Over Epochs'
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                else:
                    st.warning("⚠️ Accuracy metrics not available for the selected output type.")

            if config.OUTPUT_TYPE in ['binary', 'multiclass', 'regression']:
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

            if config.OUTPUT_TYPE == 'regression':
                # Additional Regression Metrics Plots
                if 'mae' in history_df.columns and 'val_mae' in history_df.columns:
                    fig_mae = px.line(
                        history_df,
                        y=['mae', 'val_mae'],
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
                else:
                    st.warning("⚠️ MSE metrics not available.")

        except Exception as e:
            st.error(f"❌ An error occurred during training: {e}")
