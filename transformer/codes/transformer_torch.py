import time
import math
import torch
from torch import nn

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0, f'hidden_dim must be multiple of n_heads.'
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        # in_shape: [batch_size, seq_len, hidden_dim]
        # out_shape: [batch_size, seq_len, hidden_dim]
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def split_heads(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, self.n_heads, self.head_dim)
        # [batch_size, seq_len, n_heads, head_dim]
        splits = inputs.permute(0, 2, 1, 3)
        return splits  # [batch_size, n_heads, seq_len, head_dim] -> n_heads를 앞으로

    def scaled_dot_product_attention(self, query, key, value, mask):
        key_t = key.permute(0, 1, 3, 2)
        energy = torch.matmul(query, key_t) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        attention = torch.softmax(energy, axis=-1)  # axis=-1 은 key의 문장 위치
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        return x, attention  # attention 시각화에 쓸 수 있음

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        x, attention = self.scaled_dot_product_attention(query, key, value, mask)
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]

        outputs = self.fc_o(x)
        
        return outputs, attention


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, pf_dim, hidden_dim, dropout_ratio):
        super().__init__()
        self.fc_0 = nn.Linear(hidden_dim, pf_dim)
        self.fc_1 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs):
        x = torch.relu(self.fc_0(inputs))
        x = self.dropout(x)
        outputs = self.fc_1(x)
        return outputs


class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio

        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs, mask=None):
        attn_outputs, _ = self.self_attention(inputs, inputs, inputs, mask)
        attn_outputs = self.dropout(attn_outputs)
        attn_outputs = self.self_attn_norm(inputs+attn_outputs)  # residual connection

        ff_outputs = self.pos_feedforward(attn_outputs)
        ff_outputs = self.dropout(ff_outputs)
        ff_outputs = self.pos_ff_norm(attn_outputs+ff_outputs)  # residual connection

        return ff_outputs


class DecoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.encd_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encd_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        self_attn_outputs, _ = self.self_attention(target, target, target, target_mask)
        self_attn_outputs = self.dropout(self_attn_outputs)
        self_attn_outputs = self.self_attn_norm(target+self_attn_outputs)

        encd_attn_outputs, attention = self.encd_attention(target, encd, encd, encd_mask)
        encd_attn_outputs = self.dropout(encd_attn_outputs)
        encd_attn_outputs = self.encd_attn_norm(self_attn_outputs+encd_attn_outputs)

        outputs = self.pos_feedforward(encd_attn_outputs)
        outputs = self.dropout(outputs)
        outputs = self.pos_ff_norm(outputs)

        return outputs, attention


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device, max_seq_len=100):
        # input_dim = len(vocab)
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.tok_emb = nn.Embedding(input_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.encd_stk = nn.ModuleList([
            EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        emb = self.tok_emb(x) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.encd_stk:
            outputs = layer(outputs, mask)

        return outputs


class Decoder(nn.Module):
    
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device, max_seq_len=100):
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.tok_emb = nn.Embedding(output_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.decd_stk = nn.ModuleList([
            DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        batch_size = target.shape[0]
        seq_len = target.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        emb = self.tok_emb(target) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.decd_stk:
            outputs, attention = layer(outputs, encd, target_mask, encd_mask)

        outputs = self.fc_out(outputs)

        return outputs, attention
        

class Transformer(nn.Module):

    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, n_heads, pf_dim,
                 in_seq_len, out_seq_len, pad_idx, dropout_ratio, device):
        super().__init__()
        self.device = device

        self.encoder = Encoder(
            input_dim, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device, in_seq_len
        )
        self.decoder = Decoder(
            output_dim, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device, out_seq_len
        )
        self.pad_idx = pad_idx

    def create_padding_mask(self, inputs, for_target=False):
        mask = (inputs != self.pad_idx).unsqueeze(1).unsqueeze(2)
        if for_target:
            target_len = inputs.shape[1]
            target_sub_mask = torch.tril(torch.ones((target_len, target_len), device = self.device)).bool()
            mask = mask & target_sub_mask
        return mask

    def forward(self, inp, tar):
        inp_mask = self.create_padding_mask(inp)
        tar_mask = self.create_padding_mask(tar, True)

        enc_inp = self.encoder(inp, inp_mask)
        output, attention = self.decoder(tar, enc_inp, tar_mask, inp_mask)

        return output, attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_one_epoch(model, dl, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for inp, tar in dl:
        inp, tar = inp.to(device), tar.to(device)

        optimizer.zero_grad()

        outputs, _ = model(inp, tar[:,:-1])

        output_dim = outputs.shape[-1]

        outputs = outputs.contiguous().view(-1, output_dim)
        tar = tar[:,1:].contiguous().view(-1)

        loss = criterion(outputs, tar)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)


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


def train(model, n_epochs, es_patience, train_dl, valid_dl,
          optimizer, criterion, clip, device, model_path, model_name='chatbot'):
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_dl is not None:
            if valid_loss < best_valid_loss:
                best_epoch = epoch
                print('Best!')
                best_valid_loss = valid_loss
                torch.save(model, model_path+model_name+'.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        if valid_dl is not None:
            print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')

            if epoch-best_epoch >= es_patience:
                print(f'Best Epoch: {best_epoch + 1:02}')
                print(f'\tBest Train Loss: {train_loss:.3f} | Best Train PPL: {math.exp(train_loss):.3f}')
                print(f'\tBest Validation Loss: {valid_loss:.3f} | Best Validation PPL: {math.exp(valid_loss):.3f}')
                break
    
    if valid_dl is None:
        torch.save(model, model_path+model_name+'.pt')
