import tensorflow as tf

class StackedLSTM(tf.keras.Layer):
    def __init__(self, num_layers: int, rnn_size: int, dropout: float) -> None:
        super(StackedLSTM, self).__init__()

        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Droupout(dropout)
        self.layers = [tf.keras.layers.LSTMCell(rnn_size) for i in range(num_layers)]

    def call(self, inputs, hidden):
        h0, c0 = hidden
        h1, c1 = [], []

        for i, layer in enumerate(self.layers):
            h1i, c1i = layer(inputs, (h0[i], c0[i]))
            inputs = h1i

            if i != self.num_layers:
                inputs = self.droupout(inputs)
            
            h1 += [h1i]
            c1 += [c1i]
        
        h1 = tf.stack(h1)
        c1 = tf.stack(c1)

        return inputs, (h1, c1)