import typing

from reward.Bleu import score_sentence, score_corpus

from constants.constants import EOS, UNK, PAD

#TODO: Add typing for all the shit here

def clean_up_sentence(sentence, remove_unknown = False, remove_eos = False):
    if EOS in sentence:
        sentence = sentence[:sentence.index(EOS) + 1]
    
    if remove_unknown:
        sentence = [word for word in sentence if word != UNK]
    if remove_eos:
        if len(sentence) > 0 and sentence[-1] == EOS:
            sentence = sentence[:-1]
    
    return sentence

def single_sentence_blue(hypothesis, reference):
    length = len(hypothesis)

    hypothesis = clean_up_sentence(hypothesis)
    reference = clean_up_sentence(reference)
    
    hypothesis_length = len(hypothesis)

    if hypothesis_length == 0:
        score = 0
        hypothesis = [PAD] * length
    else:
        score = score_sentence(hypothesis, reference, 4, 1)[-1]
        while len(hypothesis) < length:
            hypothesis.append(PAD)

    return score, hypothesis

def sentence_bleu(hypotheses: typing.List[int], references: typing.List[int]) -> typing.Tuple[typing.List[float], typing.List[int]]:
    return zip(*map(lambda pair: single_sentence_blue(pair[0], pair[1]), zip(hypotheses, references)))

def corpus_bleu(hypotheses, references):
    assert(len(hypotheses) == len(references))

    cleaned_hypotheses = [clean_up_sentence(hypothesis, remove_unknown = False, remove_eos = True) for hypothesis in hypotheses]
    cleaned_references = [clean_up_sentence(reference, remove_unknown = False, remove_eos = True) for reference in references]

    return score_corpus(cleaned_hypotheses, cleaned_references, 4)