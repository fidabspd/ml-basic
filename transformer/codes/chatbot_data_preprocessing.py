import re
import numpy as np


def add_space(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def to_tokens(sentence, tokenizer, to_ids=True):
    if to_ids:
        tokens = tokenizer.encode(sentence).ids
        vocab_size = tokenizer.get_vocab_size()
        start_token, end_token = vocab_size, vocab_size+1
        tokens = [start_token]+tokens+[end_token]
    else:
        tokens = tokenizer.encode(sentence).tokens
    return tokens


def pad_seq(seq, tokenizer, max_seq_len):
    pad_token = tokenizer.encode('[PAD]').ids[0]
    padded_seq = seq+[pad_token]*(max_seq_len-len(seq))
    return padded_seq


def preprocess_sentence(sentence, tokenizer, max_seq_len):
    sentence = add_space(sentence)
    sentence = to_tokens(sentence, tokenizer)
    sentence = pad_seq(sentence, tokenizer, max_seq_len)
    return sentence


def preprocess_sentences(sentences, tokenizer, max_seq_len):
    prep =  list(map(
        lambda sentence:
            preprocess_sentence(sentence, tokenizer, max_seq_len),
        sentences
    ))
    return np.array(prep)
