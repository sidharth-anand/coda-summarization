import tensorflow as tf

#TODO: This shit iis probably wrong
class Generator(tf.keras.Model):
    def __init__(self, target_vocabulary_size: int) -> None:
        print(target_vocabulary_size)
        super(Generator, self).__init__()

        self.target_vocabulary_size = target_vocabulary_size
        self.linear = tf.keras.layers.Dense(target_vocabulary_size, use_bias=True)

    #TODO: this shit is wrong as well
    def call(self, inputs):
        x = tf.reshape(inputs, [-1, inputs.shape[-1]])
        print(x.shape)
        return self.linear(tf.reshape(inputs, [-1, inputs.shape[-1]]))

    #TODO: is this shit really correct?
    def predict(self, inputs):
        logits = self.call(inputs)
        predictions = tf.math.argmax(logits, axis=-1)

        return predictions