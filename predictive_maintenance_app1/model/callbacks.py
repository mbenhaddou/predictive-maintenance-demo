# callbacks.py

import tensorflow as tf

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, progress_bar, status_text, output_type='binary'):
        """
        Initializes the StreamlitCallback.

        Args:
            epochs (int): Total number of training epochs.
            progress_bar (streamlit.delta_generator.DeltaGenerator): Streamlit progress bar object.
            status_text (streamlit.delta_generator.DeltaGenerator): Streamlit text object for status updates.
            output_type (str): Type of output. Options: 'binary', 'multiclass', 'regression'.
        """
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_epoch = 0
        self.output_type = output_type.lower()

        # Define metric keys based on task_type
        if self.output_type in ['binary', 'multiclass']:
            self.metric_keys = {
                'accuracy': 'accuracy',
                'val_accuracy': 'val_accuracy',
                'loss': 'loss',
                'val_loss': 'val_loss'
            }
        elif self.output_type == 'regression':
            self.metric_keys = {
                'mae': 'mae',
                'mse': 'mse',
                'val_mae': 'val_mae',
                'val_mse': 'val_mse',
                'loss': 'loss',
                'val_loss': 'val_loss'
            }
        else:
            raise ValueError("Unsupported task_type. Choose from 'binary', 'multiclass', 'regression'.")

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch during training.

        Args:
            epoch (int): Current epoch index.
            logs (dict): Metric results for this epoch.
        """
        self.current_epoch += 1

        # Initialize a message string
        message = f"Epoch {self.current_epoch}/{self.epochs}\n"

        # Extract and append relevant metrics to the message
        for metric, key in self.metric_keys.items():
            value = logs.get(key)
            if value is not None:
                message += f"{metric.upper()}: {value:.4f}\n"

        # Update the progress bar
        self.progress_bar.progress(self.current_epoch / self.epochs)

        # Update the status text
        self.status_text.text(message)
