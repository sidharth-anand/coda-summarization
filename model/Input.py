import tensorflow as tf

class InputLayer(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super(InputLayer, self).__init__()

        self.input_layer = tf.keras.layers.Input(shape=(None,), dtype=(), name='input_layer')