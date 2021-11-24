import tensorflow as tf

class StackedLSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, rnn_size: int, dropout: float) -> None:
        super(StackedLSTM, self).__init__()

        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layers = [tf.keras.layers.LSTMCell(rnn_size) for _ in range(num_layers)]

    def call(self, inputs, hidden):
        print(inputs.shape)
        print(hidden[0].shape, hidden[1].shape)

        h0, c0 = hidden

        print('h0',type(h0))
        print('c0',type(c0))
        
        print(h0.shape)
        print(c0.shape)

        h1, c1 = [], []

        for i, layer in enumerate(self.layers):
            _, state = layer(inputs, states=(h0[i], c0[i]))
            h1i, c1i = state[0], state[1]

            print(h1i)
            print(c1i)
            print(h1i.shape)
            print(c1i.shape)
            inputs = h1i

            if not i == self.num_layers:
                inputs = self.dropout(inputs)
            
            h1 += [h1i]
            c1 += [c1i]
        
        h1 = tf.stack(h1)
        c1 = tf.stack(c1)

        print('shit over')

        return inputs, (h1, c1)