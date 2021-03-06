import tensorflow as tf

from model.Hybrid2Seq import Hybrid2Seq

from constants.constants import PAD

class Evaluator:
    def __init__(self, model: Hybrid2Seq, metrics, dictionaries, max_length: int = 50):
        self.model = model
        self.dictionaries = dictionaries

        self.cross_entropy_loss = metrics['cross_entropy_loss']
        self.sentence_reward = metrics['sentence_reward']
        self.corpus_reward = metrics['corpus_reward']

        self.max_length = max_length

    def evaluate(self, data, pred_file = None):
        total_loss = 0
        total_words = 0
        total_sentences = 0
        total_sentence_reward = 0

        all_predictions = []
        all_targets = []
        all_sources = []

        for i in range(len(data)):
            batch = data[i]
            batch = batch[0]

            targets = batch[2]

            code_attention_mask = tf.cast(tf.math.equal(batch[1][2][0], tf.constant(PAD)), dtype=tf.float32)
            text_attention_mask = tf.cast(tf.math.equal(batch[0][0], tf.constant(PAD)), dtype=tf.float32)

            self.model.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)

            outputs = self.model(batch, training=False)

            loss_weights = tf.cast(tf.math.not_equal(targets, tf.constant(PAD)), dtype=tf.float32)
            num_words = tf.reduce_sum(loss_weights)

            prediction = self.model.predict(outputs)
            loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(tf.cast(targets, dtype=tf.float32), tf.cast(prediction, dtype=tf.float32), loss_weights))
            
            predictions = self.model.translate(batch, self.max_length)
            sources = batch[0][0]

            rewards, _ = self.sentence_reward(predictions.numpy().tolist(), targets.numpy().tolist())

            all_predictions.extend(predictions.numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_sources.extend(sources.numpy().tolist())

            total_loss += loss
            total_words += num_words
            total_sentence_reward += tf.reduce_sum(rewards, axis=-1)
            total_sentences += batch[2].shape[1]

        loss = total_loss / total_words
        sentence_reward = total_sentence_reward / total_sentences
        corpus_reward = self.corpus_reward(all_predictions, all_targets)

        return loss, sentence_reward, corpus_reward
