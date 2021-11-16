import tensorflow as tf

class BinaryTreeLeafModule(tf.keras.Layer):
    def __init__(self, hidden_state_size: int) -> None:
        super(BinaryTreeLeafModule, self).__init__()
        self.cx = tf.keras.layers.Dense(hidden_state_size, use_bias=True)
        self.ox = tf.keras.layers.Dense(hidden_state_size, use_bias=True)

    def call(self, inputs):
        c = self.cx(inputs)
        output = tf.keras.activations.sigmoid(inputs)
        hidden =  tf.math.multiply(output, tf.math.tanh(c))
        
        return hidden, (c, hidden)