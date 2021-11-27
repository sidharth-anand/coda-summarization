import tensorflow as tf

# TODO: This shit iis probably wrong


class Generator(tf.keras.Model):
    def __init__(self, target_vocabulary_size: int) -> None:
        print(target_vocabulary_size)
        super(Generator, self).__init__()

        self.target_vocabulary_size = target_vocabulary_size
        self.linear = tf.keras.layers.Dense(
            target_vocabulary_size, use_bias=True)

    # TODO: this shit is wrong as well - nope
    def call(self, inputs):
        outputs = []

        for i in range(inputs.shape[1]):
            current_word = inputs[:, i, :]
            current_word = tf.squeeze(current_word)
            outputs.append(self.linear(current_word))

        return tf.stack(outputs, axis=1)

    #input: (B, H)
    def translate(self, inputs):
        return self.linear(inputs)

    # TODO: is this shit really correct? - yes. correct
    def predict(self, inputs):
        logits = self.call(inputs)
        predictions = tf.math.argmax(logits, axis=-1)

        return predictions
