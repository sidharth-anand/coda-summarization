import math
import time

import tensorflow as tf

from train.Evaluator import Evaluator

from constants.constants import PAD

class ReinforceTrainer:
    def __init__(self, actor, critic, training_data, validation_data, metrics, dictonaries, actor_optimizer, critic_optimizer, max_length: int, reinforcement_learning_rate: float, no_update: bool = False):
        self.actor = actor
        self.critic = critic

        self.training_data = training_data
        self.validation_data = validation_data
        self.evalualtor = Evaluator(actor, metrics, dictonaries, max_length)

        self.actor_loss_function = metrics['cross_entropy_loss']
        self.critic_loss_function = metrics['critic_loss']
        self.sentence_reward_function = metrics['sentence_reward']

        self.dictionaries = dictonaries

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.max_length = max_length
        self.reinforcement_learning_rate = reinforcement_learning_rate
        self.no_update = no_update
        #TODO: Add reward shaping here

    def train(self, start_epoch: int, end_epoch: int, pretrain_critic: bool):
        start_time = time.time()

        self.actor_optimizer.last_loss = self.critic_optimizer.last_loss
        self.actor_optimizer.set_learning_rate(self.reinforcement_learning_rate)

        if pretrain_critic:
            self.critic_optimizer.set_learning_rate(1e-3)
        else:   
            self.critic_optimizer.set_learning_rate(self.reinforcement_learning_rate)

        for epoch in range(start_epoch, end_epoch + 1):
            print('* REINFORCE epoch *')
            print(f'Actor optimizer LearningRate: {self.actor_optimizer.learning_rate}')
            print(f'Critic optimizer LearningRate: {self.critic_optimizer.learning_rate}')

            if pretrain_critic:
                print('Pretraining Critic')
            
            no_update = self.no_update and (not pretrain_critic) and (not epoch == start_epoch)
            if no_update:
                print('No update')

            training_reward, critic_loss = self.train_epoch(epoch, pretrain_critic, no_update, start_time)
            print(f'Training sentence reward: {training_reward * 100}')
            print(f'Critic loss: {critic_loss}')

            validation_loss, validation_sentence_reward, validation_corpus_reward = self.evalualtor.evaluate(self.validation_data)
            validation_perplexity = math.exp(min(validation_loss, 100))

            print(f'Validation perplexity: {validation_perplexity}')
            print(f'Validation sentence reward: {validation_sentence_reward * 100}')
            print(f'Validation corpus reward: {validation_corpus_reward * 100}')

            if no_update:
                break
                
            self.actor_optimizer.update_learning_rate(-validation_sentence_reward, epoch)

            if not pretrain_critic:
                self.critic_optimizer.set_learning_rate(self.actor_optimizer.learning_rate)

            #TODO: Checkpoint the model here
        
    def train_epoch(self, epoch_index: int, pretrain_critic: bool, no_update: bool, start_time):
        self.actor.train()

        total_reward = 0
        reported_reward = 0

        total_critic_loss = 0
        reported_critic_loss = 0

        total_sentences = 0
        reported_sentences = 0

        total_words = 0
        reported_words = 0

        last_reported_time = time.time()

        for i in range(len(self.training_data)):
            with tf.GradientTape() as actor_tape:
                with tf.GradientTape() as critic_tape:
                    batch = self.training_data[i]

                    targets = batch[2]
                    code_attention_mask = tf.math.equal(batch[1][2][0], tf.constant(PAD))
                    text_attention_mask = tf.math.equal(batch[0][0], tf.constant(PAD))

                    batch_size = targets.shape[1]

                    #TODO: these are pt methods. change to tf equivalents
                    self.actor.zero_grad()
                    self.critic.zero_grad()

                    self.model.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)

                    samples, outputs = self.actor.sample(batch, self.max_length)

                    rewards, samples = self.sentence_reward_function(samples, targets)
                    total_reward = tf.reduce_sum(rewards)

                    #TODO: add reward shaping here

                    samples = tf.Variable(samples)
                    rewards = tf.Variable(rewards)

                    critic_loss_weights = tf.math.not_equal(samples, tf.constant(PAD, dtype=tf.float64))
                    num_words = tf.reduce_sum(critic_loss_weights)

                    if not no_update:
                        baselines = self.critic((batch[0], batch[1], samples, batch[3]), eval=False, regression=True)
                        
                        critic_loss = tf.nn.weighted_cross_entropy_with_logits(baselines, rewards, critic_loss_weights)
                        grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                        self.critic.optimizer.apply_gradients(grads, self.critic.trainable_variables)
                    else:
                        critic_loss = 0

                    if not pretrain_critic and not no_update:
                        normalized_rewards = tf.Variable(rewards - baselines)
                        actor_loss_weights = tf.math.mul(normalized_rewards, critic_loss_weights)

                        actor_loss = tf.nn.weighted_cross_entropy_with_logits(outputs, samples, actor_loss_weights)
                        grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                        self.actor.optimizer.apply_gradients(grads, self.actor.trainable_variables)
                        
                        self.actor_optimizer.step()
                    else:
                        actor_loss = 0

                    
            
            total_reward += total_reward
            reported_reward += rewards

            total_sentences += batch_size
            reported_sentences += batch_size

            total_critic_loss += critic_loss
            reported_critic_loss += critic_loss

            total_words += num_words
            reported_words += num_words

            print(f'Iteration: {i}, loss: {actor_loss}')
            print(f'Iteration: {i}, reward: {reported_reward / reported_sentences}')

            if i % 100 == 0 and i > 0:
                print(f'Epoch {epoch_index} {i}/{len(self.training_data)} --- Actor Reward: {reported_reward * 100 / reported_sentences} --- Critic Loss: {reported_critic_loss / reported_words} --- {reported_words / (time.time() - last_reported_time)}tokens/sec --- {time.time() - start_time} seconds from start')

                reported_reward = 0
                reported_sentences = 0
                reported_critic_loss = 0
                reported_words = 0

                last_reported_time = time.time()

            return total_reward / total_sentences, total_critic_loss / total_words
            

