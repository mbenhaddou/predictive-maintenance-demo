import numpy as np

class SequenceGenerator:
    """
    Class for generating sequences and labels for LSTM input.
    """

    def __init__(self, df, sequence_length, sequence_cols, output_column):
        self.df = df
        self.sequence_length = sequence_length
        self.sequence_cols = sequence_cols
        self.output_column = output_column

    def generate_sequences(self, padding_strategy='zero'):
        """
        Generates sequences and labels for LSTM input.
        """
        seq_array, label_array = [], []
        for id in self.df['id'].unique():
            id_df = self.df[self.df['id'] == id]
            seq_gen = self._gen_sequence(id_df, padding_strategy=padding_strategy)
            seq_array.extend(seq_gen)
            labels = self._gen_labels(id_df, padding_strategy=padding_strategy)
            label_array.extend(labels)
        return np.array(seq_array), np.array(label_array)

    def _gen_sequence(self, id_df, padding_strategy=None):
        """
        Generates sequences for a single engine (id), optionally with padding.
        """
        data_array = id_df[self.sequence_cols].values
        num_elements = data_array.shape[0]
        sequences = []
        num_features = data_array.shape[1]
        padding_length = self.sequence_length - 1

        if padding_strategy == 'zero':
            padding = np.zeros((padding_length, num_features))
            padded_data = np.vstack((padding, data_array))
        elif padding_strategy == 'mean':
            padding_value = np.mean(data_array, axis=0) if num_elements > 0 else np.zeros(num_features)
            padding = np.tile(padding_value, (padding_length, 1))
            padded_data = np.vstack((padding, data_array))
        elif padding_strategy is None:
            # No padding
            padded_data = data_array
        else:
            raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

        if padding_strategy is None:
            # Generate sequences without padding
            for i in range(num_elements - self.sequence_length + 1):
                sequence = data_array[i:i + self.sequence_length, :]
                sequences.append(sequence)
        else:
            # Generate sequences with padding
            for i in range(num_elements):
                sequence = padded_data[i:i + self.sequence_length, :]
                sequences.append(sequence)

        return sequences

    def _gen_labels(self, id_df, padding_strategy=None):
        """
        Generates labels for the sequences.
        """
        labels = id_df[self.output_column].values
        if padding_strategy is None:
            # Adjust labels to match sequences without padding
            labels = labels[self.sequence_length - 1:]
        return labels
