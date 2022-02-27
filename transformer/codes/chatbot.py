import argparse

import pickle

import torch

from chatbot_data_preprocessing import *
from transformer_torch import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--ans_seq_len', type=str, help='Enter the answer sequence length.', default=50)
    parser.add_argument('--stop_sign', type=str, help='Stop sign to stop conversation.', default='q')

    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='chatbot')
    parser.add_argument('--tokenizer_path', type=str, default='../model/')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer')

    return parser.parse_args()


def qna(question, transformer, tokenizer, ans_seq_len, device):

    transformer.eval()

    vocab_size = tokenizer.get_vocab_size()
    start_token, end_token = vocab_size, vocab_size+1
    
    question_tokens = to_tokens(question, tokenizer)
    question_tokens = torch.LongTensor(question_tokens).unsqueeze(0).to(device)
    question_mask = transformer.create_padding_mask(question_tokens)
    with torch.no_grad():
        question_encd = transformer.encoder(question_tokens, question_mask)

    output_tokens = [start_token]

    for _ in range(ans_seq_len):
        target_tokens = torch.LongTensor(output_tokens).unsqueeze(0).to(device)

        target_mask = transformer.create_padding_mask(target_tokens, True)
        with torch.no_grad():
            output, attention = transformer.decoder(target_tokens, question_encd, target_mask, question_mask)

        pred_token = output.argmax(2)[:,-1].item()
        output_tokens.append(pred_token)

        if pred_token == end_token:
            break
            
    output_sentence = tokenizer.decode(output_tokens)
    
    return output_sentence, attention


def main(args):

    ANS_SEQ_LEN = args.ans_seq_len
    STOP_SIGN = args.stop_sign
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(TOKENIZER_PATH+TOKENIZER_NAME+'.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')

    print('Start ChatBot\nEnter the message\n')
    print(f'To stop conversation, Enter "{STOP_SIGN}"\n')
    while True:
        question = input('You: ')
        if question == STOP_SIGN:
            break
        answer, _ = qna(question, transformer, tokenizer, ANS_SEQ_LEN, device)
        print(f'ChatBot: {answer}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
