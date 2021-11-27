def weighted_mse(logits, targets, weights):
    losses = (logits - targets)**2
    losses = losses * weights
    return losses
