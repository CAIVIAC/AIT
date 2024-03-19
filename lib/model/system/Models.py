''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
# from system.Layers import EncoderLayer, DecoderLayer

from model.system.Layers import EncoderLayer, DecoderLayer
import model.modules.cells as C




def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, channel, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).type(torch.uint8)
        # torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
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
        """
        :param x [bs, N, C=512]
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        """
        - n_src_vocab = 9521
        - d_word_vec = 512
        - n_layers = 6
        - n_head = 8
        - d_k, d_v = 64, 64
        - d_model = 512
        - d_inner = 2048
        - pad_idx = 1
        - n_position = 200
        """

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        :param src_seq
            - size: [bs, N_s=32] (Transformer)
            - size: [bs, N_s, C] (Proposal)
        :param src_mask
            - size: [bs, 1, N_s=32]
        """

        enc_slf_attn_list = []

        # -- Forward
        # self.src_word_em(src_seq) --> [bs, N_s, C=512]

        # enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq))) # [bs, N_s=32, C=512]
        enc_output = self.dropout(self.position_enc(src_seq)) # [bs, N_s=32, C=512]
        enc_output = self.layer_norm(enc_output) # [bs, N_s=32, C=512]

        for enc_idx, enc_layer in enumerate(self.layer_stack):
            """
            :var enc_output [bs, N_s, C=512]
            :var enc_slf_attn [bs, n_head=8, N_s, N_s]
            """
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        """
        - n_trg_vocab = 9521
        - d_word_vec = 512
        - n_layers = 6
        - n_head = 8
        - d_k, d_v = 64, 64
        - d_model = 512
        - d_inner = 2048
        - pad_idx = 1
        - n_position = 200
        """

        # self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        """
        :param trg_seq
            - size: [bs, N_t=29]
        :param trg_mask
            - size: [bs, N_t=29, N_t=29]
        """

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq))) # [bs, N_t=29, C=512]
        dec_output = self.dropout(self.position_enc(trg_seq)) # [bs, N_t=29, C=512]
        dec_output = self.layer_norm(dec_output) # [bs, N_t=29, C=512]

        for dec_idx, dec_layer in enumerate(self.layer_stack):
            """
            :var dec_output [bs, N_t=29, C=512]
            :var dec_slf_attn_list [bs, n_head, N_t=29, N_t=29]
            :var dec_enc_attn [bs, n_head=8, N_t=29, N_s]
            """
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
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.channels = d_word_vec
        self.enc_emb = nn.Sequential(
            C.conv2d_1x1(d_word_vec*2, d_word_vec, bias=True),
        )
        self.dec_emb = nn.Sequential(
            C.conv2d_1x1(d_word_vec*2, d_word_vec, bias=True),
        )

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        self.dec_trans = nn.Sequential(
            C.conv2d_1x1(d_word_vec, d_word_vec*2, bias=True),
        )

        # FC project d_model=512 --> n_trg_vocab=9521
        # self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # self.x_logit_scale = 1.
        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        #     self.x_logit_scale = (d_model ** -0.5) #0.04419417382415922
        # 
        # if emb_src_trg_weight_sharing:
        #     self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, x_props, x_query):
        """
        :param x_props: proposal features
            - size: [bs*num_props, C=1024, H=7, W=7] (Proposal)
        :param x_query: query features
            - size: [bs, C=1024, H=8, W=8] (Query)
        """
        bp, c_props, h_props, w_props = x_props.size() # bp = bs * num_props
        bs, c_query, h_query, w_query = x_query.size()
        device = x_props.device

        num_props = int(bp // bs)

        # ====================== Dimension Deduction ======================  #
        # Reduce dimesion to fit the hyperparamter settings of attention-is-all-you-need
        x_props = self.enc_emb(x_props) # [bs*props, C=512, H=7, W=7]
        x_query = self.dec_emb(x_query) # [bs, C=512, H=8, W=8]

        """ For each proposal having query feature """
        r_query = x_query.unsqueeze(1).repeat(1, num_props, 1, 1, 1) # [bs, num_props, C=512, H=7, W=7]

        src_seq = x_props.view(bp, self.channels, -1) # Proposal: [bs * num_props, C=512, T=HW=49]
        trg_seq = r_query.view(bp, self.channels, -1) # Query:    [bs * num_props, C=512, T=HW=64]

        n_s, n_t = src_seq.size(-1), trg_seq.size(-1)
        src_seq, trg_seq = src_seq.permute(0, 2, 1), trg_seq.permute(0, 2, 1) # [bp, T, C=512]

        src_mask = torch.ones((bp, 1, n_s), dtype=torch.uint8).to(device) # False: padded words
        pad_mask = torch.zeros((bp, 1, n_t-n_s), dtype=torch.uint8).to(device) # False: padded words
        src_mask = torch.cat((src_mask, pad_mask), dim=2) # [bs, 1, N_t]

        trg_mask = torch.ones((bp, 1, n_t), dtype=torch.uint8).to(device) # False: padded words
        trg_mask = trg_mask & get_subsequent_mask(trg_mask) # [bs, N_t, N_t]

        # src_mask = get_pad_mask(src_seq, self.src_pad_idx) # [bs, 1, N_s]
        # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq) # [bs, N_t, N_t]

        # pad zeros
        zpadings = torch.zeros((bp, n_t-n_s, self.channels)).to(device)
        src_seq = torch.cat((src_seq, zpadings), dim=1) # padding zeros: [bp, N_t, C=512]

        enc_output, *_ = self.encoder(src_seq, src_mask) # [bp, N_t, C=512]
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask) # [bp, N_t, C=512]

        # ====================== Dimension Transform ======================  #
        dec_output = dec_output.permute(0, 2, 1).contiguous() # [bp, C=512, N_t]
        dec_output = dec_output.view(bp, self.channels, h_query, w_query) # [bp, C=512, H=8, W=8]
        dec_output = self.dec_trans(dec_output) # [bp, C=1024, H=8, W=8]

        return dec_output

