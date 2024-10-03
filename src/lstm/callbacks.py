# In your predictive_maintenance.py file or Streamlit app
import tensorflow as tf

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        # Update progress bar
        self.progress_bar.progress(self.current_epoch / self.epochs)
        # Update status text
        self.status_text.text(f"Epoch {self.current_epoch}/{self.epochs}\n"
                              f"Loss: {loss:.4f} - Accuracy: {acc:.4f}\n"
                              f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")