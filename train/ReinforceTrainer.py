import math
import time
import numpy as np
import os

import tensorflow as tf

from train.Evaluator import Evaluator

from loss.Loss import weighted_mse

from constants.constants import PAD

class ReinforceTrainer:
    def __init__(self, actor, critic, training_data, validation_data, metrics, dictonaries, max_length: int, reinforcement_learning_rate: float, no_update: bool = False):
        self.actor = actor
        self.critic = critic

        self.training_data = training_data
        self.validation_data = validation_data
        self.evalualtor = Evaluator(actor, metrics, dictonaries, max_length)

        self.actor_loss_function = metrics['cross_entropy_loss']
        self.critic_loss_function = metrics['critic_loss']
        self.sentence_reward_function = metrics['sentence_reward']

        self.dictionaries = dictonaries

        self.max_length = max_length
        self.reinforcement_learning_rate = reinforcement_learning_rate
        self.no_update = no_update
        #TODO: Add reward shaping here

    def train(self, start_epoch: int, end_epoch: int, pretrain_critic: bool, resume:bool):
        start_time = time.time()

        self.actor.optimizer.lr.assign(self.reinforcement_learning_rate)

        if pretrain_critic:
            self.critic.optimizer.lr.assign(1e-3)
        else:   
            self.critic.optimizer.lr.assign(self.reinforcement_learning_rate)

        for epoch in range(start_epoch, end_epoch + 1):

            if resume and os.path.isfile(f'weights/critic.h5'):
                if pretrain_critic:
                    self.model.load_weights(f'weights/critic.h5')
                elif not pretrain_critic and os.path.isfile(f'weights/actor.h5'):
                    self.model.load_weights(f'weights/critic.h5')
                    self.model.load_weights(f'weights/actor.h5')
                continue

            print('* REINFORCE epoch *')
            print(f'Actor optimizer LearningRate: {self.actor.optimizer.lr.read_value()}')
            print(f'Critic optimizer LearningRate: {self.critic.optimizer.lr.read_value()}')

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

            val_batch = self.validation_data[0]

            
            #TODO: prettify this clusterfuck

            val_batch = val_batch[0]
            targets = val_batch[2].numpy()[0:5]
            source = val_batch[0][0].numpy()[0:5]
            code_attention_mask = tf.cast(tf.math.equal(val_batch[1][2][0], tf.constant(PAD)), dtype=tf.float32)
            text_attention_mask = tf.cast(tf.math.equal(val_batch[0][0], tf.constant(PAD)), dtype=tf.float32)
            self.actor.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)
            #actor results
            outputs = self.actor(val_batch, regression=True)
            outputs = tf.argmax(outputs, axis=-1).numpy()[0:5]
            with open('results/RL_actor_results.txt','a') as f:
                for i,sentence in enumerate(outputs):
                    f.write(f'code {i+1}' + ' '.join(self.dictionaries['src'].reverse_lookup(token) for token in source[i]) + '\n')
                    f.write(f'prediction {i+1}: ' + ' '.join(self.dictionaries['tgt'].reverse_lookup(token) for token in sentence) + '\n')
                    f.write(f'target {i+1}: ' + ' '.join(self.dictionaries['tgt'].reverse_lookup(token) for token in targets[i]) + '\n')
            f.close()

            #critic results
            outputs = self.critic(val_batch, regression=True)
            outputs = tf.argmax(outputs, axis=-1).numpy()[0:5]
            with open('results/RL_critic_results.txt','a') as f:
                for i,sentence in enumerate(outputs):
                    f.write(f'code {i+1}' + ' '.join(self.dictionaries['src'].reverse_lookup(token) for token in source[i]) + '\n')
                    f.write(f'prediction {i+1}: ' + ' '.join(self.dictionaries['tgt'].reverse_lookup(token) for token in sentence) + '\n')
                    f.write(f'target {i+1}: ' + ' '.join(self.dictionaries['tgt'].reverse_lookup(token) for token in targets[i]) + '\n')
            f.close()

            if no_update:
                break
            
            if not pretrain_critic:
                self.critic.optimizer.lr.assign(self.actor.optimizer.lr.read_value())

            if pretrain_critic:
                if not resume or (resume and not os.path.isfile(f'weights/critic.h5')):
                    print(f'Saving epoch {epoch} of Critic:')
                    self.critic.save_weights(f'weights/critic.h5')

            else:
                if not resume or (resume and not (os.path.isfile(f'weights/critic.h5') and os.path.isfile(f'weights/actor.h5'))):
                    print(f'Saving epoch {epoch} of Critic:')
                    self.critic.save_weights(f'weights/critic.h5')
                    print(f'Saving epoch {epoch} of Actor:')
                    self.actor.save_weights(f'weights/actor.h5')

        
    def train_epoch(self, epoch_index: int, pretrain_critic: bool, no_update: bool, start_time):
        total_rewards = 0
        reported_rewards = 0

        total_critic_loss = 0
        reported_critic_loss = 0

        total_sentences = 0
        reported_sentences = 0

        total_words = 0
        reported_words = 0

        last_reported_time = time.time()

        for i in range(len(self.training_data)):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                batch = self.training_data[i]
                batch = batch[0]

                targets = batch[2]
                one_hot_target = batch[4]
                
                code_attention_mask = tf.cast(tf.math.equal(batch[1][2][0], tf.constant(PAD)), dtype=tf.float32)
                text_attention_mask = tf.cast(tf.math.equal(batch[0][0], tf.constant(PAD)), dtype=tf.float32)

                batch_size = targets.shape[1]

                self.actor.hybrid_decoder.attention.apply_mask(code_attention_mask, text_attention_mask)

                samples, outputs = self.actor.sample(batch, self.max_length)

                rewards, samples = self.sentence_reward_function(samples.numpy().tolist(), targets.numpy().tolist())
                total_reward = tf.reduce_sum(rewards)

                #TODO: add reward shaping here

                samples = tf.Variable(tf.convert_to_tensor(samples), trainable=True, name='actor:samples')
                rewards = tf.Variable(tf.stack([rewards] * one_hot_target.shape[2], axis=1), trainable=True, name='actor:rewards')

                critic_loss_weights = tf.cast(tf.math.not_equal(samples, tf.constant(PAD)), dtype=tf.float32)
                num_words = tf.reduce_sum(critic_loss_weights)

                critic_loss_weights = tf.ones((critic_loss_weights.shape[0], critic_loss_weights.shape[1], one_hot_target.shape[2]), dtype=tf.float32) * tf.expand_dims(critic_loss_weights, axis=-1)

                if not no_update:
                    baselines = self.critic((batch[0], batch[1], samples, batch[3], batch[4]), regression=True)
                    
                    rewards = tf.expand_dims(rewards, axis=-1)
                    rewards = tf.broadcast_to(rewards, [rewards.shape[0], rewards.shape[1], baselines.shape[1]])
                    rewards = tf.reshape(rewards, [rewards.shape[0], rewards.shape[2], rewards.shape[1]])

                    mask = np.zeros(rewards.shape, dtype=np.float32)
                    mask[:, rewards.shape[1] - 1, :] = tf.squeeze(tf.one_hot(samples, depth=one_hot_target.shape[2])[:, -1, :]).numpy()
                    rewards = rewards * mask

                    critic_loss = weighted_mse(baselines, rewards, critic_loss_weights)
                   
                    grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

                    critic_loss = tf.reduce_sum(critic_loss)
                else:
                    critic_loss = 0

                if not pretrain_critic and not no_update:
                    samples = tf.Variable(tf.one_hot(samples, one_hot_target.shape[2]), trainable=True)

                    normalized_rewards = tf.Variable(rewards - baselines, trainable=True)
                    actor_loss_weights = tf.math.multiply(normalized_rewards, critic_loss_weights)
                    
                    predictions = self.actor.generator(outputs)

                    actor_loss = tf.nn.weighted_cross_entropy_with_logits(samples, predictions, pos_weight=actor_loss_weights)
                    grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                    
                    self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                else:
                    actor_loss = 0

            total_rewards += total_reward
            reported_rewards += total_reward

            total_sentences += batch_size
            reported_sentences += batch_size

            total_critic_loss += critic_loss
            reported_critic_loss += critic_loss

            total_words += num_words
            reported_words += num_words

            print(f'Iteration: {i}, loss: {actor_loss}')
            print(f'Iteration: {i}, reward: {reported_rewards / reported_sentences}')

            if i % 1 == 0 and i > 0:
                print(f'Epoch {epoch_index} {i}/{len(self.training_data)} --- Actor Reward: {reported_rewards * 100 / reported_sentences} --- Critic Loss: {reported_critic_loss / reported_words} --- {reported_words / (time.time() - last_reported_time)}tokens/sec --- {time.time() - start_time} seconds from start')

                reported_rewards = 0
                reported_sentences = 0
                reported_critic_loss = 0
                reported_words = 0

                last_reported_time = time.time()

        return total_rewards / total_sentences, total_critic_loss / total_words
            

