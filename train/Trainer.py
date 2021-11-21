import datetime
import math
import os
import time

import tensorflow as tf

from train.Evaluator import Evaluator

from constants.constants import PAD

class Trainer:
    def __init__(self, model, train_data_gen, validation_data_gen, metrics, dictionaries, optimizer, iterations_per_log = 100):
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
            print('* CrossEntropy Epoch *')
            print(f'Model optimizer LearningRate: {self.optimizer.learning_rate}')

            train_loss = self.train_epoch(epoch, start_time)

            print(f'Train perplexity: {math.exp(min(train_loss, 100))}')

            validation_loss, validation_sentence_reward, validation_corpus_reward = self.evaluator.evaluate(self.validation_data_gen)
            validation_perplexity = math.exp(min(validation_loss, 100))

            print(f'Validation perplexity: {validation_perplexity}')
            print(f'Validation sentence reward: {validation_sentence_reward * 100}')
            print(f'Validation sentence reward: {validation_corpus_reward * 100}')

            self.optimizer.update_learning_rate(validation_loss, epoch)

            #TODO: checkpoint the model here

    def train_epoch(self, epoch_index: int, training_start):
        # self.model.train()

        total_loss = 0
        reported_loss = 0

        total_words = 0
        reported_words = 0

        last_reported_time = time.time()

        for i in range(len(self.train_data_gen)):
            batch = self.train_data_gen[i]

            targets = batch[2]
            code_attention_mask = tf.math.equal(batch[1][2][0], tf.constant(PAD))
            text_attention_mask = tf.math.equal(batch[0][0], tf.constant(PAD))

            self.model.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)

            outputs = self.model(batch, False)

            loss_weights = tf.math.not_equal(targets, tf.constant(PAD, dtype=tf.float64))
            num_words = tf.reduce_sum(loss_weights)
            loss = self.model.backwards(outputs, targets, loss_weights)

            reported_loss += loss
            total_loss += loss

            reported_words += num_words
            total_words += num_words

            if i % self.iterations_per_log == 0 and i > 0:
                print(f'Epoch {epoch_index} --- {i}/{len(self.train_data_gen)} batches --- perplexity: {math.exp(reported_loss / reported_words)} --- {time.time() - last_reported_time} tokens/second --- {time.time() - training_start}s since training start')

                reported_loss = 0
                reported_words = 0
                last_reported_time = time.time()
            
        return total_loss / total_words
            
