import tensorflow as tf

#TODO: This shit iis probably wrong
class Generator(tf.keras.Model):
    def __init__(self, target_vocabulary_size: int) -> None:
        super(Generator, self).__init__()

        self.target_vocabulary_size = target_vocabulary_size
        self.linear = tf.keras.layers.Dense(target_vocabulary_size, use_bias=True)

    def call(self, inputs):
        return self.linear(inputs)

    #TODO: is this shit really correct?
    def predict(self, inputs):
        logits = self.call(inputs)
        predictions = tf.math.argmax(logits, axis=-1)

        return predictions