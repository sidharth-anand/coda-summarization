import time
import math

import tensorflow as tf

from train.Evaluator import Evaluator

from constants.constants import PAD


class Trainer:
    def __init__(self, model: tf.keras.Model, train_data_gen: tf.keras.Sequential, validation_data_gen: tf.keras.Sequential, metrics: dict, dictionaries: dict, optimizer: tf.keras.optimizers, iterations_per_log=1):
        self.model = model
        self.train_data_gen = train_data_gen
        self.validation_data_gen = validation_data_gen
        self.evaluator = Evaluator(model, metrics, dictionaries)
        self.dictionaries = dictionaries
        self.optimizer = optimizer
        self.iterations_per_log = iterations_per_log

        self.loss_function = metrics['cross_entropy_loss']

    def train(self, start_epoch, end_epoch):
        start_time = time.time()

        for epoch in range(start_epoch, end_epoch + 1):
            print('* CorssEntropy Epoch *')
            print(
                f'Model optimizer LearningRate: {self.modal.optimizer.lr.read_value()}')

            train_loss = self.train_epoch(epoch, start_time)

            print(f'Train perplexity: {math.exp(min(train_loss, 100))}')

            validation_loss, validation_sentence_reward, validation_corpus_reward = self.evaluator.evaluate(
                self.validation_data_gen)
            validation_perplexity = math.exp(min(validation_loss, 100))

            print(f'Validation perplexity: {validation_perplexity}')
            print(
                f'Validation sentence reward: {validation_sentence_reward * 100}')
            print(
                f'Validation sentence reward: {validation_corpus_reward * 100}')

            # TODO: checkpoint the model here

    def train_epoch(self, epoch_index: int, training_start):
        total_loss = 0
        reported_loss = 0

        total_words = 0
        reported_words = 0

        last_reported_time = time.time()

        print('started train epoch')
        print(len(self.train_data_gen))

        for i in range(len(self.train_data_gen)):
            with tf.GradientTape() as tape:
                batch = self.train_data_gen[i]
                batch = batch[0]

                targets = batch[2]
                one_hot_target = batch[4]
                code_attention_mask = tf.cast(tf.math.equal(batch[1][2][0], tf.constant(PAD)), dtype=tf.float32)
                text_attention_mask = tf.cast(tf.math.equal(batch[0][0], tf.constant(PAD)), dtype=tf.float32)

                self.model.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)

                outputs = self.model(batch, regression=True)
                
                loss_weights = tf.cast(tf.math.not_equal(targets, tf.constant(PAD)), dtype=tf.float32)
                num_words = tf.reduce_sum(loss_weights)

                loss_weights = tf.ones((loss_weights.shape[0], loss_weights.shape[1], one_hot_target.shape[2]), dtype=tf.float32) * tf.expand_dims(loss_weights, axis=-1)

                loss = tf.nn.weighted_cross_entropy_with_logits(one_hot_target, outputs, loss_weights)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                reported_loss += tf.reduce_sum(loss)
                total_loss += tf.reduce_sum(loss)

                reported_words += num_words
                total_words += num_words

                if i % self.iterations_per_log == 0 and i > 0:
                    print(f'Epoch {epoch_index} --- {i}/{len(self.train_data_gen)} batches --- perplexity: {(reported_loss / reported_words)} --- {time.time() - last_reported_time} tokens/second --- {time.time() - training_start}s since training start')

                    reported_loss = 0
                    reported_words = 0
                    last_reported_time = time.time()

        return total_loss / total_words
