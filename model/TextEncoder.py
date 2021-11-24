from numpy.lib.utils import source
import tensorflow as tf


class TextEncoder(tf.keras.layers.Layer):
    def __init__(self, dictonaries, source_vocab_size: int, hidden_state_size: int = 512, word_embedding_size: int = 512, num_layers: int = 1, dropout: float = 0.3) -> None:
        super(TextEncoder, self).__init__()

        self.word_lut = tf.keras.layers.Embedding(
            source_vocab_size, word_embedding_size)
        # TODO: Add dropout here
        self.rnn = tf.keras.layers.LSTM(
            hidden_state_size, return_state=True, return_sequences=True)

        self.dictonaries = dictonaries

    def call(self, inputs):
        output, hidden_t, cell_t = self.rnn(self.word_lut(inputs[0]))

        return (hidden_t, cell_t), output
