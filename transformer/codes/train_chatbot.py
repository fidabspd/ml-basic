import os
import time
import math
import argparse
import pickle

from tokenizers import BertWordPieceTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from chatbot_data_preprocessing import *
from transformer_torch import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--file_name', type=str, default='ChatbotData')
    parser.add_argument('--graph_log_path', type=str, default='../logs/graph/')
    parser.add_argument('--tokenizer_path', type=str, default='../model/')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer')
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='chatbot')
    parser.add_argument('--train_log_path', type=str, default='../logs/train_logs/')

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_one_epoch(model, dl, optimizer, criterion, clip, device, n_check=5):

    n_data = len(dl.dataset)
    n_batch = len(dl)
    batch_size = dl.batch_size
    if n_check < 0:
        print('n_check must be larger than 0. Adjust `n_check = 0`')
        n_check = 0
    if n_batch < n_check:
        print(f'n_check should be smaller than n_batch. Adjust `n_check = {n_batch}`')
        n_check = n_batch
    if n_check:
        check = [int(n_batch/n_check*(i+1)) for i in range(n_check)]
    train_loss = 0

    model.train()
    for b, (inp, tar) in enumerate(dl):
        inp, tar = inp.to(device), tar.to(device)

        outputs, _ = model(inp, tar[:,:-1])
        # decoder의 input 마지막 padding 제거, len 1 줄임.
        # 이유는 transformer의 최종 output의 shape이 [batch_size, query_len(tar), hidden_dim]이기 때문인데,
        # 우리가 예측해야하는 문장에는 <sos> 토큰이 없어야한다.
        # (예측 seq생성 과정에서 [<sos>]를 첫번째 `tar`로 넣어주고 for문을 돌릴 것이기 때문.)
        # 따라서 정답으로 쓰일 `tar`의 seq_len은 <sos> 토큰을 제외한 `len(tar)-1`이기 때문에 shape을 맞춰야 loss가 계산 가능하다.
        # 번역기로 예를들어 쉽게 말하자면
        # inp: [<sos>, hi, <eos>, <pad>, <pad>], tar: [<sos>, 안녕, <eos>, <pad>, <pad>, <pad>]을 이용해
        # [안녕, <eos>, <pad>, <pad>, <pad>, <pad>]의 예측값을 만들어내야한다.

        output_dim = outputs.shape[-1]
        outputs = outputs.contiguous().view(-1, output_dim)
        tar = tar[:,1:].contiguous().view(-1)  # loss 계산할 정답으로 쓰일 `tar`는 <sos> 토큰 제거
        loss = criterion(outputs, tar)
        train_loss += loss.item()/n_data

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if n_check and b+1 in check:
            n_data_check = b*batch_size + len(inp)
            train_loss_check = train_loss*n_data/n_data_check
            print(f'loss: {train_loss_check:>10f}  [{n_data_check:>5d}/{n_data:>5d}]')

    return train_loss


def evaluate(model, dl, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for inp, tar in dl:
            inp, tar = inp.to(device), tar.to(device)
            outputs, _ = model(inp, tar[:,:-1])

            output_dim = outputs.shape[-1]

            outputs = outputs.contiguous().view(-1, output_dim)
            tar = tar[:,1:].contiguous().view(-1)
            loss = criterion(outputs, tar)

            epoch_loss += loss.item()

    return epoch_loss / len(dl)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, n_epochs, es_patience, train_dl, valid_dl, optimizer,
          criterion, clip, device, model_path, train_log_path, model_name='chatbot'):
    if train_log_path is not None:
        writer = SummaryWriter(train_log_path)
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        print('-'*30, f'\nEpoch: {epoch+1:02}', sep='')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        if train_log_path is not None:
            writer.add_scalar('train loss', train_loss, epoch)
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion, device)
            if train_log_path is not None:
                writer.add_scalar('valid loss', valid_loss, epoch)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_dl is not None:
            if valid_loss < best_valid_loss:
                best_epoch = epoch
                print('Best!')
                best_valid_loss = valid_loss
                torch.save(model, model_path+model_name+'.pt')

        print(f'Train Loss: {train_loss:.3f}\nEpoch Time: {epoch_mins}m {epoch_secs}s')
        if valid_dl is not None:
            print(f'Validation Loss: {valid_loss:.3f}')

            if epoch-best_epoch >= es_patience:
                print(f'\nBest Epoch: {best_epoch+1:02}')
                print(f'\tBest Train Loss: {train_loss:.3f}')
                print(f'\tBest Validation Loss: {valid_loss:.3f}')
                break
    
    if train_log_path is not None:
        writer.close()
    if valid_dl is None:
        torch.save(model, model_path+model_name+'.pt')


def main(args):

    DATA_PATH = args.data_path
    FILE_NAME = args.file_name
    GRAPH_LOG_PATH = args.graph_log_path
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    TRAIN_LOG_PATH = args.train_log_path

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
    f = open(DATA_PATH+FILE_NAME+'.txt', 'r')
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
        files = DATA_PATH+FILE_NAME+'.txt',
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

    # Train model
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_IDX)

    if not VALIDATE:
        valid_dl = None
    train(
        transformer, N_EPOCHS, ES_PATIENCE, train_dl, valid_dl,
        optimizer, criterion, CLIP, device, MODEL_PATH, TRAIN_LOG_PATH, MODEL_NAME
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
