import re

import numpy as np

from torch.utils.data import Dataset


def add_space(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)  # 기호 띄어쓰기
    sentence = sentence.strip()
    return sentence


def to_tokens(sentence, tokenizer, to_ids=True):
    if to_ids:
        tokens = tokenizer.encode(sentence).ids
    else:
        tokens = tokenizer.encode(sentence).tokens
    return tokens


def pad_seq(seq, tokenizer, max_seq_len):
    pad_token = tokenizer.encode('[PAD]').ids[0]
    vocab_size = tokenizer.get_vocab_size()
    start_token, end_token = vocab_size, vocab_size+1
    padded_seq = [start_token]+seq+[end_token]+[pad_token]*(max_seq_len-len(seq)-2)
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


class ChatBotDataset(Dataset):
    def __init__(self, questions, answers):
        assert len(questions) == len(answers)
        self.questions = questions
        self.answers = answers
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question, answer = self.questions[idx], self.answers[idx]
        return question, answer
