import tensorflow as tf

from CodeEncoder import CodeEncoder
from TextEncoder import TextEncoder
from HybridDecoder import HybridDecoder
from Generator import Generator

from constants.constants import BOS, EOS

#TODO: figure out backprop and loss
class Hybrid2Seq(tf.keras.Model):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int) -> None:
        super(Hybrid2Seq, self).__init__()

        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size

        self.code_encoder = CodeEncoder(source_vocabulary_size)
        self.text_encoder = TextEncoder(target_vocabulary_size)
        self.hybrid_decoder = HybridDecoder(target_vocabulary_size)
        self.generator = Generator(target_vocabulary_size)
    
    #TODO: tf is context
    def initialize_decoder_output(self, context):
        batch_size = context.shape[0]
        hidden_size = (batch_size, self.hybrid_decoder.hidden_size)
        #TODO: Return shit here

    #TODO: add typing to this shit
    #TODO: eval???
    def initialize(self, inputs, eval: bool):
        target = inputs[2]
        trees = inputs[1][0]
        lengths = inputs[1][1]
        source_text = inputs[0]
        
        tree_encoder_context_padded = []
        tree_encoder_hidden_0 = []
        tree_encoder_hidden_1 = []

        for tree in trees:
            text_encoder_context, tree_encoder_hidden = self.code_encoder(tree, lengths)
            tree_encoder_context_padded.append(text_encoder_context)
            tree_encoder_hidden_0.append(tree_encoder_hidden[0])
            tree_encoder_hidden_1.append(tree_encoder_hidden[1])

        tree_encoder_context_padded = tf.stack(tree_encoder_context_padded, axis=1)
        tree_encoder_hidden = (tf.stack(tree_encoder_hidden_0, axis=1), tf.stack(tree_encoder_hidden_1, axis=1))

        text_encoder_hidden, text_encoder_context = self.text_encoder(source_text)
        initial_output = self.initialize_decoder_output(text_encoder_context)

        initial_token = tf.Variable(tf.Tensor([BOS] * initial_output.shape[0], dtype=tf.int32))
        embedding = self.hybrid_decoder.word_lut(initial_token)

        return target, (embedding, initial_output, tree_encoder_hidden, tf.transpose(tree_encoder_context_padded, [0, 1]), text_encoder_hidden, tf.transpose(text_encoder_context, [0, 1]))

    #TODO: add typing to this shit
    def call(self, inputs, eval, training=False):
        targets, initial_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, initial_states)

        if training:
            logits = self.generator(outputs)
            return tf.reshape(logits, targets.shape)
        
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
            output, hidden = self.hybrid_decoder.step(embeddings, output, hidden, context)
            logit = self.generator(output)
            prediction = tf.reshape(logit.max(1)[1], -1)
            predictions.append(prediction)

            eos_indices |= (prediction == EOS)
            if eos_indices.sum() == batch_size:
                break

            #TODO: fix this    
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
            output, hidden = self.decoder.step(embeddings, output, hidden, context)
            outputs.append(output)

            distance = tf.keras.layers.softmax()(self.generator(output))
            #TODO: tf should go here?
            sample = 1 #???
            samples.append(sample)

            eos_indices |= (sample == EOS)
            if eos_indices.sum() == batch_size:
                break

            #TODO: fix this too
            embeddings = self.hybrid_decoder.word_lut(tf.Tensor(sample))

        outputs = tf.stack(outputs, axis=1)
        samples = tf.stack(samples, axis=1)

        return samples, outputs