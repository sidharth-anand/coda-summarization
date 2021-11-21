import datetime
import math
import os
import time

import tensorflow as tf
from tensorflow.keras import optimizers

from train.Evaluator import Evaluator

from constants.constants import PAD

class Trainer:
    def __init__(self, model: tf.keras.Model, train_data_gen: tf.keras.Sequential, validation_data_gen: tf.keras.Sequential, metrics: dict, dictionaries: dict, optimizer: tf.keras.optimizers, iterations_per_log = 100):
        self.model = model
        self.train_data_gen = train_data_gen
        self.validation_data_gen = validation_data_gen
        self.evaluator = Evaluator(model, metrics, dictionaries)
        self.dictionaries = dictionaries
        self.optimizer = optimizer
        self.iterations_per_log = iterations_per_log

        self.loss_function = metrics['xent_loss']

    def train(self, start_epoch, end_epoch):
        
        epochs = end_epoch + 1 - start_epoch
        self.model.compile(optimizer=tf.keras.optimizers.Adam)
        self.model.fit(x=self.train_data_gen, epochs=epochs, validation_data=self.validation_data_gen)

            
