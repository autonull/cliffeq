import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import time
import urllib.request
from cliffeq.attention.geometric import CliffordAttention
from cliffeq.algebra.utils import embed_vector

class ScanDataset(Dataset):
    def __init__(self, data, word2idx_cmd, word2idx_act, max_len_cmd, max_len_act):
        self.data = data
        self.word2idx_cmd = word2idx_cmd
        self.word2idx_act = word2idx_act
        self.max_len_cmd = max_len_cmd
        self.max_len_act = max_len_act

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cmd = item['commands'].split()
        act = item['actions'].split()

        cmd_ids = [self.word2idx_cmd.get(w, self.word2idx_cmd['<UNK>']) for w in cmd]
        act_ids = [self.word2idx_act['<SOS>']] + [self.word2idx_act.get(w, self.word2idx_act['<UNK>']) for w in act] + [self.word2idx_act['<EOS>']]

        cmd_padded = cmd_ids + [self.word2idx_cmd['<PAD>']] * (self.max_len_cmd - len(cmd_ids))
        act_padded = act_ids + [self.word2idx_act['<PAD>']] * (self.max_len_act - len(act_ids))

        return torch.tensor(cmd_padded[:self.max_len_cmd]), torch.tensor(act_padded[:self.max_len_act])

def download_scan_split(split_name='length'):
    base_url = "https://raw.githubusercontent.com/brendenlake/SCAN/master/"
    if split_name == 'length':
        train_url = base_url + "length_split/tasks_train_length.txt"
        test_url = base_url + "length_split/tasks_test_length.txt"
    elif split_name == 'addprim_jump':
        train_url = base_url + "add_prim_split/tasks_train_addprim_jump.txt"
        test_url = base_url + "add_prim_split/tasks_test_addprim_jump.txt"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    def parse_file(url):
        print(f"Downloading from {url}...")
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        data = []
        for line in content.strip().split('\n'):
            if 'OUT:' in line:
                cmd_part, act_part = line.split('OUT:')
                cmd = cmd_part.replace('IN:', '').strip()
                act = act_part.strip()
                data.append({'commands': cmd, 'actions': act})
        return data

    train_data = parse_file(train_url)
    test_data = parse_file(test_url)
    return train_data, test_data

def load_scan_data(split_name='length', batch_size=64):
    print(f"Loading SCAN dataset split: {split_name}...")
    train_data, test_data = download_scan_split(split_name)

    # Build vocab
    cmds = [item['commands'].split() for item in train_data]
    acts = [item['actions'].split() for item in train_data]

    word2idx_cmd = {'<PAD>': 0, '<UNK>': 1}
    for cmd in cmds:
        for w in cmd:
            if w not in word2idx_cmd:
                word2idx_cmd[w] = len(word2idx_cmd)

    word2idx_act = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for act in acts:
        for w in act:
            if w not in word2idx_act:
                word2idx_act[w] = len(word2idx_act)

    max_len_cmd = max(len(c) for c in cmds) + 2
    max_len_act = max(len(a) for a in acts) + 5 # Extra room for SOS/EOS

    # Need to check test max lens too just in case
    test_cmds = [item['commands'].split() for item in test_data]
    test_acts = [item['actions'].split() for item in test_data]
    max_len_cmd = max(max_len_cmd, max(len(c) for c in test_cmds) + 2)
    max_len_act = max(max_len_act, max(len(a) for a in test_acts) + 5)

    train_set = ScanDataset(train_data, word2idx_cmd, word2idx_act, max_len_cmd, max_len_act)
    test_set = ScanDataset(test_data, word2idx_cmd, word2idx_act, max_len_cmd, max_len_act)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader, word2idx_cmd, word2idx_act, max_len_cmd, max_len_act

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StandardScanModel(nn.Module):
    def __init__(self, vocab_size_cmd, vocab_size_act, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embed_cmd = nn.Embedding(vocab_size_cmd, d_model)
        self.embed_act = nn.Embedding(vocab_size_act, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size_act)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoder(self.embed_cmd(src))
        tgt_emb = self.pos_encoder(self.embed_act(tgt))

        # Causal mask for decoder
        if tgt_mask is None:
            tgt_len = tgt.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        out = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(out)

    def generate(self, src, max_len, sos_idx, eos_idx, pad_idx):
        device = src.device
        src_emb = self.pos_encoder(self.embed_cmd(src))
        memory = self.transformer.encoder(src_emb)

        B = src.shape[0]
        tgt = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoder(self.embed_act(tgt))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

            tgt = torch.cat([tgt, next_token], dim=1)

            finished |= (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break

        return tgt

class CliffordTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, clifford_dim, sig_g, dim_feedforward=512, dropout=0.1, use_orientation_bias=False):
        super().__init__()
        self.self_attn = CliffordAttention(nhead, clifford_dim, sig_g, use_orientation_bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src: (B, L, d_model)
        # self_attn expects (B, L, d_model)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CliffordTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, clifford_dim, sig_g, dim_feedforward=512, dropout=0.1, use_orientation_bias=False):
        super().__init__()
        self.self_attn = CliffordAttention(nhead, clifford_dim, sig_g, use_orientation_bias)
        self.multihead_attn = CliffordAttention(nhead, clifford_dim, sig_g, use_orientation_bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class CliffordScanModel(nn.Module):
    def __init__(self, vocab_size_cmd, vocab_size_act, d_model=128, nhead=4, clifford_dim=8, sig_g=None, num_layers=2, use_orientation_bias=False):
        super().__init__()
        if sig_g is None:
            sig_g = torch.tensor([1.0, 1.0])
        self.d_model = d_model
        self.embed_cmd = nn.Embedding(vocab_size_cmd, d_model)
        self.embed_act = nn.Embedding(vocab_size_act, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = CliffordTransformerEncoderLayer(d_model, nhead, clifford_dim, sig_g, use_orientation_bias=use_orientation_bias)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = CliffordTransformerDecoderLayer(d_model, nhead, clifford_dim, sig_g, use_orientation_bias=use_orientation_bias)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size_act)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoder(self.embed_cmd(src))
        tgt_emb = self.pos_encoder(self.embed_act(tgt))

        if tgt_mask is None:
            tgt_len = tgt.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None,
                           tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.fc_out(out)

    def generate(self, src, max_len, sos_idx, eos_idx, pad_idx):
        device = src.device
        src_emb = self.pos_encoder(self.embed_cmd(src))
        memory = self.encoder(src_emb)

        B = src.shape[0]
        tgt = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoder(self.embed_act(tgt))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

            tgt = torch.cat([tgt, next_token], dim=1)

            finished |= (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break

        return tgt

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, max_len_act, sos_idx, eos_idx, pad_idx, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            B = src.shape[0]

            generated = model.generate(src, max_len_act, sos_idx, eos_idx, pad_idx)

            for i in range(B):
                pred = generated[i, 1:] # Skip SOS
                expected = tgt[i, 1:]

                # Cut at EOS
                pred_eos = (pred == eos_idx).nonzero()
                if len(pred_eos) > 0:
                    pred = pred[:pred_eos[0][0]]

                exp_eos = (expected == eos_idx).nonzero()
                if len(exp_eos) > 0:
                    expected = expected[:exp_eos[0][0]]

                if pred.shape == expected.shape and (pred == expected).all():
                    correct += 1
                total += 1
    return correct / total
