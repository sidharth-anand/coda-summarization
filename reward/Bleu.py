import math

from collections import defaultdict

#TODO: Add typing to all the shit here

def update_ngrams_count(sentence, ngrams, count):
    length = len(sentence)
    for n in range(1, ngrams + 1):
        for i in range(length - n + 1):
            ngram = tuple(sentence[i:i + n])
            count[ngram] += 1

def compute_bleu(prediction, prdiction_length, reference_length, smooth):
    log_brevity = 1 - max(1, (prdiction_length + smooth) / (reference_length + smooth))
    log_score = 0

    ngrams = len(prediction) - 1

    for n in range(1, ngrams + 1):
        if prediction[n][1] > 0:
            if prediction[n][0] == 0:
                prediction[n][0] = 1e-16
            
            log_precision = math.log((prediction[n][0] + smooth) / (prediction[n][1] + smooth))
            log_score += log_precision
    log_score /= ngrams

    return math.exp(log_score + log_brevity)

def score_sentence(hypothesis, reference, ngrams, smooth = 0):
    scores = []

    reference_count = defaultdict(int)
    update_ngrams_count(reference, ngrams, reference_count)
    
    prediction_count = defaultdict(int)

    predictions = []
    for n in range(ngrams + 1):
        predictions.append([0, 0])

    for i in range(len(hypothesis)):
        for n in range(1, ngrams + 1):
            if i - n + 1 < 0:
                continue
            
            ngram = tuple(hypothesis[i - n + 1:i + 1])
            prediction_count[ngram] += 1
            predictions[n][1] += 1

            if prediction_count[ngram] <= reference_count[ngram]:
                predictions[n][0] += 1
        
        scores.append(compute_bleu(predictions, i + 1, len(reference), smooth))

    return scores

def score_corpus(hypotheses, references, ngrams, smooth = 0):
    assert(len(hypotheses) == len(references))

    predictions = []
    for n in range(ngrams + 1):
        predictions.append([0, 0])
    
    hypothesis_length = 0
    reference_length = 0

    for hypothesis, reference in zip(hypotheses, references):
        reference_length += len(reference)
        reference_count = defaultdict(int)

        update_ngrams_count(reference, ngrams, reference_count)

        hypothesis_length += len(hypothesis)
        prediction_count = defaultdict(int)
        update_ngrams_count(hypothesis, ngrams, prediction_count)

        for k, v in prediction_count.items():
            n = len(k)
            predictions[n][0] += min(v, reference_count[k])
            predictions[n][1] += v

    return compute_bleu(predictions, hypothesis_length, reference_length, smooth)