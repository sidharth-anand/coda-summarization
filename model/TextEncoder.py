import tensorflow as tf

class TextEncoder(tf.keras.Layer):
    def __init__(self, dictonaries, source_vocab_size: int, hidden_state_size:int, word_embedding_size: int, num_layers:int = 1, dropout: float = 0.3) -> None:
        super(TextEncoder, self).__init__()

        self.word_lut = tf.keras.layers.Embedding(source_vocab_size, word_embedding_size)
        self.rnn = tf.keras.layers.LSTM(hidden_state_size, dropoout=dropout)

        self.dictonaries = dictonaries

    def call(self, inputs, hidden = None):
        embeddings = tf.keras.preprocessing.sequence.pad_sequences([self.word_lut(inputs[0]), inputs[1]], padding='post')
        output, hidden_t = self.rnn(embeddings, hidden)
        return hidden_t, output

    