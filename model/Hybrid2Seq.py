import tensorflow as tf

from data.Dictionary import Dictionary

from model.CodeEncoder import CodeEncoder
from model.TextEncoder import TextEncoder
from model.HybridDecoder import HybridDecoder
from model.Generator import Generator

from constants.constants import BOS, EOS, PAD

# TODO: figure out backprop and loss


class Hybrid2Seq(tf.keras.Model):
    def __init__(self, dictonaries: Dictionary, source_vocabulary_size: int, target_vocabulary_size: int) -> None:
        super(Hybrid2Seq, self).__init__()

        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size

        self.code_encoder = CodeEncoder(dictonaries, source_vocabulary_size)
        self.text_encoder = TextEncoder(dictonaries, source_vocabulary_size)
        self.hybrid_decoder = HybridDecoder(target_vocabulary_size)
        self.generator = Generator(target_vocabulary_size)

        self.optimizer = tf.keras.optimizers.Adam(0.01)

    def initialize_decoder_output(self, text_encoder_context):
        batch_size = text_encoder_context.shape[0]
        hidden_size = (batch_size, self.hybrid_decoder.hidden_state_size)
        return tf.Variable(tf.zeros(hidden_size), trainable=False)

    # TODO: add typing to this shit
    # TODO: eval???
    def initialize(self, batch):
        # batch format
        # return (make_tensor(batch_src),src_lengths), (batch_trees,tree_lengths, (make_tensor(batch_leafs),leaf_lengths)) , make_tensor(batch_tgt), range(len(batch_src))
        target = batch[2]
        one_hot_target = batch[4]
        trees = batch[1][0]
        lengths = batch[1][1]
        source_text = batch[0]

        tree_encoder_context_padded = []
        tree_encoder_hidden_0 = []
        tree_encoder_hidden_1 = []

        for tree in trees:
            tree_encoder_context, tree_encoder_hidden = self.code_encoder(
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

        initial_token = tf.Variable(tf.convert_to_tensor(
            [BOS] * initial_output.shape[0], dtype=tf.int32))
        embedding = self.hybrid_decoder.word_lut(initial_token)

        return target, one_hot_target, (embedding, initial_output, tree_encoder_hidden, tf.transpose(tree_encoder_context_padded, perm=[1, 0, 2]), text_encoder_hidden, tf.transpose(text_encoder_context, perm=[1, 0, 2]))

    def call(self, batch, training=False):
        targets, one_hot_target, initial_states = self.initialize(batch)
        outputs = self.hybrid_decoder(targets, initial_states)

        print('hybrid2seq call')
        print(outputs.shape)

        if training:
            logits = self.generator(outputs)
            print('logists shape', logits.shape)
            print('target shape', one_hot_target.shape)
            return tf.reshape(logits, one_hot_target.shape)

        print('reshaped successfully')

        return outputs

    #TODO: this is wrong
    def predict(self, outputs):
        return self.generate.predict(outputs)

    def translate(self, inputs, max_length: int):
        targets, initial_states = self.initialize(inputs, eval=True)
        embeddings, output, hidden, context = initial_states

        predictions = []
        batch_size = targets.shape[1]
        eos_indices = tf.zeros([batch_size], dtype=tf.int32)

        for i in range(max_length):
            output, hidden = self.hybrid_decoder.step(
                embeddings, output, hidden, context)
            logit = self.generator(output)
            prediction = tf.reshape(logit.max(1)[1], -1)
            predictions.append(prediction)

            eos_indices |= (prediction == EOS)
            if eos_indices.sum() == batch_size:
                break

            # TODO: fix this
            embeddings = self.hybrid_decoder.word_lut(tf.Tensor(prediction))

        predictions = tf.stack(predictions, axis=1)
        return predictions

    def sample(self, inputs, max_length: int):
        targets, initial_states = self.initialize(inputs, eval=False)
        embeddings, output, hidden, context = initial_states

        outputs = []
        samples = []
        batch_size = targets.shape[1]
        eos_indices = tf.zeros([batch_size], dtype=tf.int32)

        for i in range(max_length):
            output, hidden = self.decoder.step(
                embeddings, output, hidden, context)
            outputs.append(output)

            distance = tf.keras.layers.softmax()(self.generator(output))
            # TODO: tf should go here?
            sample = 1  # ???
            samples.append(sample)

            eos_indices |= (sample == EOS)
            if eos_indices.sum() == batch_size:
                break

            # TODO: fix this too
            embeddings = self.hybrid_decoder.word_lut(tf.Tensor(sample))

        outputs = tf.stack(outputs, axis=1)
        samples = tf.stack(samples, axis=1)

        return samples, outputs
