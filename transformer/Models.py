''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np

from .Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, channel, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).type(torch.uint8)
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            pad_idx: int,
            d_word_vec: int=512,
            d_model: int=512,
            d_inner: int=2048,
            n_layers: int=1,
            n_head: int=8,
            d_k: int=64,
            d_v: int=64,
            dropout: float=0.1,
            n_position: int=8*8):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.dropout(self.position_enc(src_seq)) # [bs, n_s, c=512]
        enc_output = self.layer_norm(enc_output) # [bs, n_s, c=512]

        for enc_idx, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            pad_idx: int,
            d_word_vec: int=512,
            d_model: int=512,
            d_inner: int=2048,
            n_layers: int=1,
            n_head: int=8,
            d_k: int=64,
            d_v: int=64,
            dropout: float=0.1,
            n_position: int=8*8):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(trg_seq)) # [bs, n_t, c=512]
        dec_output = self.layer_norm(dec_output) # [bs, n_t, c=512]

        for dec_idx, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            src_pad_idx: int=1,
            trg_pad_idx: int=1,
            d_word_vec: int=512,
            d_model: int=512,
            d_inner: int=2048,
            n_layers: int=1,
            n_head: int=8,
            d_k: int=64,
            d_v: int=64,
            dropout: float=0.1,
            n_position: int=8*8):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.channels = d_word_vec
        self.enc_emb = nn.Conv2d(d_word_vec*2, d_word_vec, 1, bias=True)
        self.dec_emb = nn.Conv2d(d_word_vec*2, d_word_vec, 1, bias=True)

        setting = dict(
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            n_position=n_position)

        encoder_setting, decoder_setting = setting.copy(), setting.copy()
        encoder_setting['pad_idx'], decoder_setting['pad_idx'] = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(**encoder_setting)
        self.decoder = Decoder(**decoder_setting)

        self.dec_trans = nn.Conv2d(d_word_vec, d_word_vec*2, 1, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(
            self,
            x_props, # proposal features of shape [batch_size*num_props, c=1024, h=7, w=7]
            x_query, # query features of shape [batch_size, c-1024, h=8, w=8]
        ):

        bp, c_props, h_props, w_props = x_props.shape # bp = batch_size * num_props
        bs, c_query, h_query, w_query = x_query.shape # bs = batch_size

        device = x_props.device

        num_props = int(bp // bs)

        # >> Reduce dimesion to fit the hyperparamter settings of attention-is-all-you-need
        x_props = self.enc_emb(x_props) # [bs * num_props, c=512, h=7, w=7]
        x_query = self.dec_emb(x_query) # [bs, c=512, h=8, w=8]

        # >> Each proposal equips with the query feature
        r_query = x_query.unsqueeze(1).repeat(1, num_props, 1, 1, 1) # [bs, num_props, c=512, h=7, w=7]

        src_seq = x_props.view(bp, self.channels, -1) # Proposal: [bs * num_props, c=512, n_s=hw=49]
        trg_seq = r_query.view(bp, self.channels, -1) # Query:    [bs * num_props, c=512, n_t=hw=64]

        n_s, n_t = src_seq.size(-1), trg_seq.size(-1)
        src_seq, trg_seq = src_seq.transpose(1, 2), trg_seq.transpose(1, 2) # [bp, n_*, c=512]

        src_mask = torch.ones((bp, 1, n_s), dtype=torch.uint8).to(device)
        pad_mask = torch.zeros((bp, 1, n_t-n_s), dtype=torch.uint8).to(device)
        src_mask = torch.cat((src_mask, pad_mask), dim=2) # [bs, 1, n_t]

        trg_mask = torch.ones((bp, 1, n_t), dtype=torch.uint8).to(device) # False: padded words
        trg_mask = trg_mask & get_subsequent_mask(trg_mask) # [bs, n_t, n_t]

        # >> pad zeros
        zpadings = torch.zeros((bp, n_t-n_s, self.channels)).to(device)
        src_seq = torch.cat((src_seq, zpadings), dim=1) # padding zeros: [bp, n_t, c=512]

        enc_output, *_ = self.encoder(src_seq, src_mask) # [bp, n_t, c=512]
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask) # [bp, n_t, c=512]

        # >> transform dimension
        dec_output = dec_output.transpose(1, 2).contiguous() # [bp, c=512, n_t]
        dec_output = dec_output.view(bp, self.channels, h_query, w_query) # [bp, c=512, h=8, w=8]
        dec_output = self.dec_trans(dec_output) # [bp, c=1024, h=8, w=8]

        return dec_output

