import tensorflow as tf
#Yenugu kill youself you mother bitch fucking
class StackedLSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, rnn_size: int, dropout: float) -> None:
        super(StackedLSTM, self).__init__()

        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layers = [tf.keras.layers.LSTMCell(rnn_size) for _ in range(num_layers)]

    def call(self, inputs, hidden):
        # inputs - (batch_size, concat(word_emeddings, hidden_input_size))
        # hidden params [0] - (batch_size, 1, word_embeddings)

        h0, c0 = hidden


        h1, c1 = [], []

        for i, layer in enumerate(self.layers):
            _, state = layer(inputs, states=(h0[i], c0[i]))
            h1i, c1i = state[0], state[1]

            #h1i - (batch_size, word_embeddings)
            inputs = h1i

            if not i == self.num_layers:
                inputs = self.dropout(inputs)
            
            h1 += [h1i]
            c1 += [c1i]
        
        h1 = tf.stack(h1)
        c1 = tf.stack(c1)

        return inputs, (h1, c1)