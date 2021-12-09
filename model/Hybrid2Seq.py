import tensorflow as tf

from data.Dictionary import Dictionary

from model.CodeEncoder import CodeEncoder
from model.TextEncoder import TextEncoder
from model.HybridDecoder import HybridDecoder
from model.Generator import Generator
from model.GraphSequenceLayer import GraphSequenceLayer

from constants.constants import BOS, EOS

class Hybrid2Seq(tf.keras.Model):
    def __init__(self, dictonaries: Dictionary, source_vocabulary_size: int, target_vocabulary_size: int) -> None:
        super(Hybrid2Seq, self).__init__()

        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size

        self.graph_encoder = GraphSequenceLayer()
        self.code_encoder = CodeEncoder(dictonaries, source_vocabulary_size)
        self.text_encoder = TextEncoder(dictonaries, source_vocabulary_size)
        self.hybrid_decoder = HybridDecoder(target_vocabulary_size)
        self.generator = Generator(target_vocabulary_size)

        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=10000, decay_rate=0.96)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def initialize_decoder_output(self, text_encoder_context):
        batch_size = text_encoder_context.shape[0]
        hidden_size = (batch_size, self.hybrid_decoder.hidden_state_size)
        return tf.Variable(tf.zeros(hidden_size), trainable=False)

    # TODO: add typing to this shit
    def initialize(self, batch):
        target = batch[2]
        one_hot_target = batch[4]
        trees = batch[1][0]
        lengths = batch[1][1]
        source_text = batch[0]

        tree_encoder_context_padded = []
        tree_encoder_hidden_0 = []
        tree_encoder_hidden_1 = []

        for tree in trees:
            tree_encoder_context, tree_encoder_hidden = self.graph_encoder(
                tree, lengths)

            tree_encoder_context_padded.append(tree_encoder_context)
            tree_encoder_hidden_0.append(tree_encoder_hidden[0])
            tree_encoder_hidden_1.append(tree_encoder_hidden[1])

        tree_encoder_context_padded = tf.concat(
            tree_encoder_context_padded, axis=0)

        tree_encoder_hidden = (tf.concat(tree_encoder_hidden_0, axis=0), tf.concat(
            tree_encoder_hidden_1, axis=0))

        text_encoder_hidden, text_encoder_context = self.text_encoder(
            source_text)
        text_encoder_hidden = (tf.expand_dims(
            text_encoder_hidden[0], axis=1), tf.expand_dims(text_encoder_hidden[1], axis=1))

        initial_output = self.initialize_decoder_output(text_encoder_context)

        initial_token = tf.Variable(tf.convert_to_tensor([BOS] * initial_output.shape[0], dtype=tf.int32))
        embedding = self.hybrid_decoder.word_lut(initial_token)

        return target, one_hot_target, (embedding, initial_output, tree_encoder_hidden, tf.transpose(tree_encoder_context_padded, perm=[1, 0, 2]), text_encoder_hidden, tf.transpose(text_encoder_context, perm=[1, 0, 2]))

    def call(self, batch, regression = False, predict = False):
        targets, _, initial_states = self.initialize(batch)
        outputs = self.hybrid_decoder(targets, initial_states)

        if regression:
            if not predict:
                logits = self.generator(outputs)
                return logits
            else:
                return self.predict(outputs)

        return outputs

    def predict(self, outputs):
        return self.generator.predict(outputs)

    def translate(self, inputs, max_length: int):
        targets, _, initial_states = self.initialize(inputs)
        embeddings, output, tree_hidden, tree_context, text_hidden, text_context = initial_states

        predictions = []
        batch_size = targets.shape[0]
        reached_eos = tf.zeros((batch_size), dtype=tf.bool)

        for _ in range(max_length):
            output, tree_hidden, text_hidden = self.hybrid_decoder.step(
                embeddings, output, tree_hidden, tree_context, text_hidden, text_context)

            logits = self.generator.translate(output)

            prediction = tf.math.argmax(logits, axis=1)
            print(prediction.shape)
            predictions.append(prediction)

            reached_eos |= (prediction == EOS)
            if tf.reduce_sum(tf.cast(reached_eos, dtype=tf.int32)) == batch_size:
                break

            embeddings = self.hybrid_decoder.word_lut(prediction)

        predictions = tf.stack(predictions, axis=1)
        return predictions

    def sample(self, inputs, max_length: int):
        targets, _, initial_states = self.initialize(inputs)
        embeddings, output, tree_hidden, tree_context, text_hidden, text_context = initial_states

        outputs = []
        samples = []

        batch_size = targets.shape[0]
        reached_eos = tf.zeros((batch_size), dtype=tf.bool)

        for _ in range(max_length):
            output, tree_hidden, text_hidden = self.hybrid_decoder.step(embeddings, output, tree_hidden, tree_context, text_hidden, text_context)
            outputs.append(output)

            distribution = tf.nn.softmax(self.generator.translate(output))

            sample = tf.squeeze(tf.random.categorical(logits=distribution, num_samples=1,))
            samples.append(sample)

            reached_eos |= (sample == EOS)
            if tf.reduce_sum(tf.cast(reached_eos, dtype=tf.int32)) == batch_size:
                break

            embeddings = self.hybrid_decoder.word_lut(sample)

        outputs = tf.stack(outputs, axis=1)
        samples = tf.stack(samples, axis=1)

        return samples, outputs
