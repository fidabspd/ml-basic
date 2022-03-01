import argparse
import pickle
import matplotlib.pyplot as plt
import torch
from chatbot_data_preprocessing import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--stop_sign', type=str, help='Stop sign to stop conversation.', default='q')
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='chatbot')
    parser.add_argument('--tokenizer_path', type=str, default='../model/')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer')

    return parser.parse_args()


class Chatbot():

    def __init__(self, transformer, tokenizer, device):
        self.transformer = transformer.to(device)
        self.transformer.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.call_qna = False

    def qna(self, question):

        ans_seq_len = self.transformer.out_seq_len

        vocab_size = self.tokenizer.get_vocab_size()
        start_token, end_token = vocab_size, vocab_size+1
        
        question_tokens = to_tokens(question, self.tokenizer)
        question_tokens = torch.LongTensor(question_tokens).unsqueeze(0).to(self.device)
        question_mask = self.transformer.create_padding_mask(question_tokens)
        with torch.no_grad():
            question_encd = self.transformer.encoder(question_tokens, question_mask)

        output_tokens = [start_token]

        for _ in range(ans_seq_len):
            target_tokens = torch.LongTensor(output_tokens).unsqueeze(0).to(self.device)

            target_mask = self.transformer.create_padding_mask(target_tokens, True)
            with torch.no_grad():
                output, attention = self.transformer.decoder(target_tokens, question_encd, target_mask, question_mask)

            pred_token = output.argmax(2)[:,-1].item()
            output_tokens.append(pred_token)

            if pred_token == end_token:
                break
                
        answer = self.tokenizer.decode(output_tokens)
        
        self.question = question
        self.answer = answer
        self.attention = attention
        self.call_qna = True
        
        return answer, attention

    def plot_attention_weights(self, draw_mean=False):
        if not self.call_qna:
            raise Exception('There is no `question`, `answer` and `attention`. Call `qna` first')
        question_token = to_tokens(self.question, self.tokenizer, to_ids=False)
        question_token = ['<sos>']+question_token+['<eos>']

        answer_token = to_tokens(self.answer, self.tokenizer, to_ids=False)
        answer_token = answer_token+['<eos>']

        attention = self.attention.squeeze(0)
        if draw_mean:
            attention = torch.mean(attention, dim=0, keepdim=True)
        attention = attention.cpu().detach().numpy()

        n_col = 4
        n_row = (attention.shape[0]-1)//n_col + 1
        fig = plt.figure(figsize = (n_col*6, n_row*6))
        for i in range(attention.shape[0]):
            plt.subplot(n_row, n_col, i+1)
            plt.matshow(attention[i], fignum=False)
            plt.xticks(range(len(question_token)), question_token, rotation=45)
            plt.yticks(range(len(answer_token)), answer_token)
        plt.show()


def main(args):

    STOP_SIGN = args.stop_sign
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    with open(TOKENIZER_PATH+TOKENIZER_NAME+'.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')

    chatbot = Chatbot(transformer, tokenizer, device)

    print('Start ChatBot\nEnter the message\n')
    print(f'To stop conversation, Enter "{STOP_SIGN}"\n')
    while True:
        question = input('You: ')
        if question == STOP_SIGN:
            break
        answer, _ = chatbot.qna(question)
        print(f'ChatBot: {answer}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
