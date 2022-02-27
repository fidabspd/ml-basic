import os
import math

import argparse

import pickle

from tokenizers import BertWordPieceTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from chatbot_data_preprocessing import *
from transformer_torch import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--file_name', type=str, default='ChatBotData.txt', help='**.txt')
    parser.add_argument('--graph_log_path', type=str, default='../logs/graph/')
    parser.add_argument('--tokenizer_path', type=str, default='../model/')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer')
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='chatbot')

    parser.add_argument('--que_max_seq_len', type=int, default=50)
    parser.add_argument('--ans_max_seq_len', type=int, default=50)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pf_dim', type=int, default=512)
    parser.add_argument('--dropout_ratio', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--clip', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--es_patience', type=int, default=5)
    parser.add_argument('--validate', type=bool, default=False)

    return parser.parse_args()


def create_tensorboard_graph(model, inputs, path):
    try:
        exist = bool(len(os.listdir(path)))
    except:
        exist = False
    if not exist:
        writer = SummaryWriter(path)
        writer.add_graph(model, inputs)
        writer.close()
        print('Saved model graph')
    else:
        print('graph already exists')


def main(args):

    DATA_PATH = args.data_path
    FILE_NAME = args.file_name
    GRAPH_LOG_PATH = args.graph_log_path
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name

    QUE_MAX_SEQ_LEN = args.que_max_seq_len
    ANS_MAX_SEQ_LEN = args.ans_max_seq_len
    N_LAYERS = args.n_layers
    HIDDEN_DIM = args.hidden_dim
    N_HEADS = args.n_heads
    PF_DIM = args.pf_dim
    DROPOUT_RATIO = args.dropout_ratio

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    CLIP = args.clip
    N_EPOCHS = args.n_epochs
    ES_PATIENCE = args.es_patience
    VALIDATE = args.validate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Load data
    questions = []
    answers = []
    f = open(DATA_PATH+FILE_NAME, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question, answer = line.split('\t')
        questions.append(question)
        answers.append(answer)
    f.close()

    # Train tokenizer
    tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
    tokenizer.train(
        files = DATA_PATH + FILE_NAME,
        vocab_size = 32000,
        min_frequency = 3,
        limit_alphabet = 6000
    )
    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    with open(TOKENIZER_PATH+TOKENIZER_NAME+'.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    vocab_size = tokenizer.get_vocab_size()
    PAD_IDX = tokenizer.encode('[PAD]').ids[0]
    INPUT_DIM = vocab_size+2  # start_token, end_token
    OUTPUT_DIM = vocab_size+2

    # Preprocess data
    questions_prep = preprocess_sentences(questions, tokenizer, QUE_MAX_SEQ_LEN)
    answers_prep = preprocess_sentences(answers, tokenizer, ANS_MAX_SEQ_LEN)

    if VALIDATE:
        train_q, valid_q = questions_prep[:-3000], questions_prep[-3000:]
        train_a, valid_a = answers_prep[:-3000], answers_prep[-3000:]
        valid_ds = ChatBotDataset(valid_q, valid_a)
        valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    else:
        train_q = questions_prep
        train_a = answers_prep

    train_ds = ChatBotDataset(train_q, train_a)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Set model
    transformer = Transformer(
        INPUT_DIM, OUTPUT_DIM, N_LAYERS, HIDDEN_DIM, N_HEADS, PF_DIM,
        QUE_MAX_SEQ_LEN, ANS_MAX_SEQ_LEN, PAD_IDX, DROPOUT_RATIO, device
    ).to(device)

    print(f'# of trainable parameters: {count_parameters(transformer):,}')
    transformer.apply(initialize_weights)

    inp, tar = iter(train_dl).next()
    inp, tar = inp.to(device), tar.to(device)
    create_tensorboard_graph(transformer, (inp, tar), GRAPH_LOG_PATH)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if not VALIDATE:
        valid_dl = None
    train(
        transformer, N_EPOCHS, ES_PATIENCE, train_dl, valid_dl,
        optimizer, criterion, CLIP, device, MODEL_PATH, MODEL_NAME
    )

    if VALIDATE:
        transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')
        valid_loss = evaluate(transformer, valid_dl, criterion, device)
        print(f'Valid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):.3f}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
