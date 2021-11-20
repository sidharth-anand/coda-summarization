import tensorflow as tf

def weighted_mse(logits, targets, weights):
    losses = (logits - targets)**2
    losses = losses * weights
    return tf.reduce_sum(losses)
