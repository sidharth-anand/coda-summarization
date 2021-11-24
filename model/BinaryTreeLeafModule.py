import tensorflow as tf

class BinaryTreeLeafModule(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size: int) -> None:
        super(BinaryTreeLeafModule, self).__init__()

        self.cx = tf.keras.layers.Dense(hidden_state_size, use_bias=True)
        self.ox = tf.keras.layers.Dense(hidden_state_size, use_bias=True)

    def call(self, input):
        c = self.cx(input)

        output = tf.keras.activations.sigmoid(self.ox(input))
        hidden =  tf.math.multiply(output, tf.math.tanh(c))
        
        return hidden, (c, hidden)