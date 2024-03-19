import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.modules.cells as C

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))

class CoAttention(nn.Module):
    def __init__(self, **kwargs):
        super(CoAttention, self).__init__()

        self.in_ch = kwargs.get('in_ch', 1024)
        self.c_hidden = kwargs.get('c_hidden', 512)
        self.with_residual = kwargs.get('with_residual', True)
        self.normlization = kwargs.get('normlization', 'division')

        """ project to embedded space """
        self.emb = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)
        self.rho = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)
        self.phi = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)

        self.omega = nn.Sequential(
            C.conv2d_1x1(in_ch=self.c_hidden, out_ch=self.in_ch),
            # C.conv2d_3x3(in_ch=self.c_hidden, out_ch=self.in_ch),
            # C.bn2d(self.in_ch),
            C.gn(32, self.in_ch),
        )
        self.theta = nn.Sequential(
            C.conv2d_1x1(in_ch=self.c_hidden, out_ch=self.in_ch),
            # C.conv2d_3x3(in_ch=self.c_hidden, out_ch=self.in_ch),
            # C.bn2d(self.in_ch),
            C.gn(32, self.in_ch),
        )

        if self.normlization == 'softmax':
            self.softmax = nn.Softmax(dim=2)

        """ Initialization """
        self.reset_params()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_qry):
        """
        :param x_img: image_feat
            - size: [bz, C=1024, H_i, W_i]
        :param x_qry: query_feat
            - size : [bz, C=1024, H_q=8, W_q=8]
        """

        if self.with_residual:
            identity_img = x_img
            identity_qry = x_qry

        bz, c_i, h_i, w_i = x_img.shape
        bz, c_q, h_q, w_q = x_qry.shape

        # ==================== Find query image similar object ==================== #
        """
        self.emb, self.phi, self.rho: Conv2d_1x1: in_ch --> self.c_hidden
        """
        emb_img = self.emb(x_img).view(bz, self.c_hidden, -1)
        emb_img = emb_img.permute(0, 2, 1).contiguous() # [bz, N_i=H_i*W_i, C=512]

        emb_qry = self.emb(x_qry).view(bz, self.c_hidden, -1)
        emb_qry = emb_qry.permute(0, 2, 1).contiguous() # [bz, N_q=H_q*W_q=64, C=512]

        rho_qry = self.rho(x_qry).view(bz, self.c_hidden, -1)
        rho_qry = rho_qry.permute(0, 2, 1) # [bz, N_q=64, C=512]

        phi_img = self.phi(x_img).view(bz, self.c_hidden, -1) # [bz, C=512, N_i]

        co_attention = torch.matmul(rho_qry, phi_img) # [bz, N_q=64, N_i=N]

        N_q, N_i = co_attention.size(1), co_attention.size(2)

        q2i_relation = co_attention
        i2q_relation = co_attention.permute(0, 2, 1).contiguous()

        if self.normlization == 'softmax':
            q2i_relation = self.softmax(q2i_relation) # [bz, N_q=64, N_i=N]
            i2q_relation = self.softmax(i2q_relation) # [bz, N_i=N, N_q=64]
        elif self.normlization == 'division':
            q2i_relation = q2i_relation / N_i # [bz, N_q=64, N_i=N]
            i2q_relation = i2q_relation / N_q # [bz, N_i=N, N_q=64]

        """ Obtain non-local feature
        - F(I) = non_img = non-local feature of Input Image
        - F(Q) = non_qry = non-local feature of Query Image
        """
        non_img = torch.matmul(i2q_relation, emb_qry) # [bz, N_i, C=512]
        non_img = non_img.permute(0, 2, 1).contiguous() # [bz, C=512, N_i]
        non_img = non_img.view(bz, self.c_hidden, h_i, w_i) # [bz, C=512, H_i, W_i]
        non_img = self.theta(non_img) # [bz, C=1024, H_i, W_i]
        if self.with_residual:
            non_img = non_img + identity_img # non-local residual: [bz, C=1024, H_i, W_i] = F(I)

        non_qry = torch.matmul(q2i_relation, emb_img) # [bz, N_q=64, C=512]
        non_qry = non_qry.permute(0, 2, 1).contiguous() # [bz, C=512, N_q=64]
        non_qry = non_qry.view(bz, self.c_hidden, h_q, w_q) # [bz, C=512, H_q=8, W_q=8]
        non_qry = self.omega(non_qry) # [bz, C=1024, H_q=8, W_q=8]
        if self.with_residual:
            non_qry = non_qry + identity_qry # non-local residual: [bz, C=1024, H_q=8, W_q=8] = F(Q)

        return non_img, non_qry

class ChAttention(nn.Module):
    def __init__(self, **kwargs):
        super(ChAttention, self).__init__()

        self.in_ch = kwargs.get('in_ch', 1024)
        self.c_hidden = kwargs.get('c_hidden', 512)
        self.with_residual = kwargs.get('with_residual', True)
        self.normlization = kwargs.get('normlization', 'softmax')

        """ project to embedded space """
        self.emb = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)
        self.rho = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)
        self.phi = C.conv2d_1x1(in_ch=self.in_ch, out_ch=self.c_hidden)

        self.omega = nn.Sequential(
            C.conv2d_1x1(in_ch=self.c_hidden, out_ch=self.in_ch),
            C.bn2d(self.in_ch),
        )
        self.theta = nn.Sequential(
            C.conv2d_1x1(in_ch=self.c_hidden, out_ch=self.in_ch),
            C.bn2d(self.in_ch),
        )
        # self.conv = nn.Sequential(
        #     C.conv2d_1x1(2 * self.in_ch, self.in_ch),
        #     nn.ReLU(inplace=True), # 1.0.12 (here)
        #     nn.Sigmoid(), # 1.0.11, 1.0.13
        #     # nn.Tanh(),
        # )

        if self.normlization == 'softmax':
            self.softmax = nn.Softmax(dim=2)

        """ Initialization """
        self.reset_params()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_qry):
        """
        :param x_img: resized image-level feature (after co-attention)
            - size : [bz, C=1024, H_i, W_i]
        :param x_qry: query-image-level feature (after co-attention)
            - size : [bz, C=1024, H_q=8, W_q=8]
        """
        if self.with_residual:
            identity_img = x_img
            identity_qry = x_qry

        bz, c_i, h_i, w_i = x_img.shape
        bz, c_q, h_q, w_q = x_qry.shape

        # ==================== Find query image similar object ==================== #
        """
        self.emb, self.phi, self.rho: Conv2d_1x1: in_ch --> self.c_hidden
        """
        emb_img = self.emb(x_img).view(bz, self.c_hidden, -1) # [bz, C=512, HW]
        # emb_img = emb_img.permute(0, 2, 1).contiguous() # [bz, HW, C=512]

        emb_qry = self.emb(x_qry).view(bz, self.c_hidden, -1) # [bz, C=512, HW]
        # emb_qry = emb_qry.permute(0, 2, 1).contiguous() # [bz, HW, C=512]

        rho_qry = self.rho(x_qry).view(bz, self.c_hidden, -1) # [bz, C=512, HW]
        # rho_qry = rho_qry.permute(0, 2, 1) # [bz, HW, C=512]

        phi_img = self.phi(x_img).view(bz, self.c_hidden, -1) # [bz, C=512, HW]

        ch_relation = torch.matmul(phi_img, rho_qry.permute(0, 2, 1).contiguous()) # [bz, C_i, C_q]

        N_i, N_q = ch_relation.size(1), ch_relation.size(2)

        i2q_relation = ch_relation
        q2i_relation = ch_relation.permute(0, 2, 1).contiguous()

        if self.normlization == 'softmax':
            i2q_relation = self.softmax(i2q_relation) # [bz, C_i, C_q]
            q2i_relation = self.softmax(q2i_relation) # [bz, C_q, C_i]
        elif self.normlization == 'division':
            i2q_relation = i2q_relation / N_q # [bz, C_i, C_q]
            q2i_relation = q2i_relation / N_i # [bz, C_q, C_i]

        """ Obtain non-local feature
        - F(I) = non_img = non-local feature of Input Image
        - F(Q) = non_qry = non-local feature of Query Image
        """
        non_img = torch.matmul(i2q_relation, emb_qry) # [bz, C_i, HW(Q)]
        # non_img = non_img.permute(0, 2, 1).contiguous() # [bz, C=512, N_i]
        non_img = non_img.view(bz, self.c_hidden, h_i, w_i) # [bz, C=512, H_i, W_i]
        non_img = self.theta(non_img) # [bz, C=1024, H_i, W_i]
        if self.with_residual:
            non_img = non_img + identity_img # non-local residual: [bz, C=1024, H_i, W_i] = F(I)

        non_qry = torch.matmul(q2i_relation, emb_img) # [bz, C_q, HW(I)]
        # non_qry = non_qry.permute(0, 2, 1).contiguous() # [bz, C=512, N_q=64]
        non_qry = non_qry.view(bz, self.c_hidden, h_q, w_q) # [bz, C=512, H_q=8, W_q=8]
        non_qry = self.omega(non_qry) # [bz, C=1024, H_q=8, W_q=8]
        if self.with_residual:
            non_qry = non_qry + identity_qry # non-local residual: [bz, C=1024, H_q=8, W_q=8] = F(Q)

        # x_img_qry = torch.cat((non_img, non_qry), dim=1) # [bz, C=2048, H, W]
        # # 1.0.11 and 1.0.13
        # gate = self.conv(x_img_qry) # [bz, C=1024, H, W]
        # v_feat = non_img + non_img * gate
        # # 1.0.12
        # # v_feat = self.conv(x_img_qry) # [bz, C=1024, H, W]

        # return v_feat
        return non_img
        # return non_qry

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, poolings=['average', 'maximum']):
        super(ChannelAttention, self).__init__()
        self.poolings = poolings
        self.channels = channels
        self.mlp = nn.Sequential(
            C.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c_x, h_x, w_x = x.size()
        c_att = None
        for ptype in self.poolings:
            if ptype == 'average':
                x_pool = F.avg_pool2d(x, (h_x, w_x), stride=(h_x, w_x)) # [bs, C, 1, 1]
            elif ptype == 'maximum':
                x_pool = F.max_pool2d(x, (h_x, w_x), stride=(h_x, w_x)) # [bs, C, 1, 1]
            elif ptype == 'lp':
                x_pool = F.lp_pool2d(x, 2, (h_x, w_x), stride=(h_x, w_x))
            elif ptype == 'logsumexp':
                # LSE pool only
                x_pool = logsumexp_2d(x)

            c_wgt = self.mlp(x_pool)

            if c_att is None:
                c_att = c_wgt
            else:
                c_att = c_att + c_wgt

        scale = self.sigmoid(c_att).view(bs, self.channels, 1, 1) # [bs, C, 1, 1]
        return scale

class MultiChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, poolings=['average', 'maximum'], n_repeat=3, **kwargs):
        super(MultiChannelAttention, self).__init__()
        self.poolings = poolings
        self.channels = channels
        self.n_repeat = n_repeat
        self.reference_dealer = kwargs.get('reference_dealer', None)
        if self.reference_dealer == 'concatenation':
            self.trans = nn.Sequential(
                C.conv2d_1x1(2 * channels, channels),
                nn.ReLU(),
            )
        self.conv = nn.Sequential(
            C.conv1d_1x1(channels, channels // reduction),
            nn.ReLU(),
            C.conv1d_1x1(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_ref=None):
        bz, c_x, h_x, w_x = x.size()
        if x_ref is not None:
            bz, c_r, h_r, w_r = x_ref.size()
        c_att = None
        x_vec = None
        for ptype in self.poolings: # multi-representation
            if ptype == 'average':
                x_pool = F.avg_pool2d(x, (h_x, w_x), stride=(h_x, w_x)) # [bz, C, 1, 1]
                if x_ref is not None:
                    r_pool = F.avg_pool2d(x_ref, (h_r, w_r), stride=(h_r, w_r)) # [bz, C, 1, 1]
            elif ptype == 'maximum':
                x_pool = F.max_pool2d(x, (h_x, w_x), stride=(h_x, w_x)) # [bz, C, 1, 1]
                if x_ref is not None:
                    r_pool = F.max_pool2d(x_ref, (h_r, w_r), stride=(h_r, w_r)) # [bz, C, 1, 1]
            elif ptype == 'lp':
                x_pool = F.lp_pool2d(x, 2, (h_x, w_x), stride=(h_x, w_x))
            elif ptype == 'logsumexp':
                # LSE pool only
                x_pool = logsumexp_2d(x)

            if x_ref is not None:
                if self.reference_dealer == 'addition':
                    x_pool = x_pool + r_pool
                elif self.reference_dealer == 'concatenation':
                    x_pool = torch.cat((x_pool, r_pool), dim=1) # [bz, 2C, 1, 1]
                    x_pool = self.trans(x_pool) # [bz, C, 1, 1]

            x_pool = x_pool.view(bz, self.channels, 1) # [bz, C, N=1]
            x_pool = x_pool.repeat(1, 1, self.n_repeat) # [bz, C, N]
            c_wgt = self.conv(x_pool) # [bz, C, N]

            if c_att is None:
                c_att = c_wgt
                x_vec = x_pool
            else:
                c_att = c_att + c_wgt
                x_vec = x_vec + x_pool

        scale = self.sigmoid(c_att)#.view(bz, self.channels, 1) # [bz, C, 1]
        x_exc = scale * x_vec # excited x: [bz, C, N]
        return scale, x_exc

class SKChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, poolings=['average', 'maximum'], **kwargs):
        super(SKChannelAttention, self).__init__()
        self.poolings = poolings
        self.channels = channels
        self.num_exc = kwargs.get('num_exc', 3)
        self.num_states = kwargs.get('num_states', 2)
        self.h_qry = kwargs.get('h_qry', 8)
        self.w_qry = kwargs.get('w_qry', 8)

        """ Selective Kernel Operations (SK) """
        self.convs = nn.ModuleList([nn.Sequential(
            C.conv2d_3x3(in_ch=2 * channels, out_ch=channels, dilation=d),
            C.bn2d(channels),
            nn.ReLU(inplace=True),
            ) for d in range(1, self.num_states+1)
        ])
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            C.fc(channels, channels // reduction),
            C.bn1d(channels // reduction),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            C.fc(channels // reduction, channels)\
            for _ in range(self.num_states)
        ])
        self.softmax = nn.Softmax(dim=1)

        """ K-Excitations Operations """
        self.kexc_conv = nn.Sequential(
            C.conv1d_1x1(self.h_qry * self.w_qry, self.num_exc),
            nn.ReLU(inplace=True),
        )

        """ Squeeze Operations """
        self.adp_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1) if ptype == 'average' else\
            nn.AdaptiveMaxPool1d(1) if ptype == 'maximum' else\
            nn.LPPool1d(norm_type=2, kernel_size=self.h_qry)
            for ptype in poolings
        ])

        """ Excitation Operations """
        self.mlp = nn.Sequential(
            C.conv1d_1x1(channels, channels // reduction),
            nn.ReLU(),
            C.conv1d_1x1(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()

        self.reset_params()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # if m.bias is not None:
            #     fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            #     bound = 1 / math.sqrt(fan_in)
            #     nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_qry):
        bz, c_img, h_img, w_img = x_img.size()
        bz, c_qry, h_qry, w_qry = x_qry.size()

        # ======== Selective Kernel Operations ======= #
        x_img_bar = F.interpolate(x_img, size=(h_qry, w_qry), mode='bilinear', align_corners=False)
        x_img_qry = torch.cat((x_img_bar, x_qry), dim=1) # [bz, C=2048, H=8, W=8]

        u_feat_mat = [self.convs[ns](x_img_qry) for ns in range(self.num_states)]
        u_feat_mat = torch.stack(u_feat_mat).permute(1, 0, 2, 3, 4) # [bz, N_s=num_states, C=1024, H=8, W=8]

        u_feat = torch.sum(u_feat_mat, dim=1) # [bz, C=1024, H=8, W=8]
        s_feat = self.se_pool(u_feat).view(bz, self.channels) # [bz, C=1024]
        z_feat = self.se_fc(s_feat) # [bz, C'=C / reduction]

        vector = [self.fcs[ns](z_feat) for ns in range(self.num_states)]
        vector = torch.stack(vector).permute(1, 0, 2) # [bz, N_s=num_states, C=1024]
        vector = self.softmax(vector) # [bz, N_s=num_states, C=1024]
        vector = vector.view(bz, self.num_states, self.channels, 1, 1) # [bz, N_s, C=1024, H=1, W=1]

        v_feat = (u_feat_mat * vector.expand_as(u_feat_mat)).sum(dim=1) # [bz, C=1024, H=8, W=8]

        # ======== K-Excitations Operations ======= #
        """ From h_qry * w_qry image pixels to num_K excitations (still image feature) for proposals """
        k_feat = v_feat.view(bz, self.channels, -1).permute(0, 2, 1) # [bz, HW=64, C=1024]
        k_feat = self.kexc_conv(k_feat) # [bz, num_K, W=C=1024]
        k_feat = k_feat.permute(0, 2, 1) # [bz, C=1024, N_k=num_K]

        c_att = None
        x_vec = None
        for pid, pool in enumerate(self.adp_pool):
            x_pool = pool(k_feat) # [bz, C=1024, 1]
            c_wgt = self.mlp(x_pool) # [bz, C=1024, 1]

            if c_att is None:
                c_att = c_wgt # [bz, C=1024, 1]
                x_vec = x_pool # [bz, C=1024, 1]
            else:
                c_att = c_att + c_wgt # [bz, C=1024, 1]
                x_vec = x_vec + x_pool # [bz, C=1024, 1]

        scale = self.sigmoid(c_att) # [bz, C, 1]
        x_exc = k_feat * scale.expand_as(k_feat) # excited query-image-level feature: [bz, C, N_k]

        return scale, x_exc

class DiverseAttention(nn.Module):
    def __init__(self, channels, reduction=16, num_exc=3, poolings=['average', 'maximum'], **kwargs):
        super(DiverseAttention, self).__init__()

        self.num_exc = num_exc
        self.channels = channels
        self.h_qry = kwargs.get('h_qry', 8)
        self.w_qry = kwargs.get('w_qry', 8)
        self.modality_id = kwargs.get('modality_id', 0)
        self.interpolation = kwargs.get('interpolation', 'bilinear')

        self.fusion_dict = {
            0: 'selective_kernel', 1: 'conv3x3', 2: 'convlstm',
        }

        """ Feature Aggregation """
        self.aggregator = FeatureFusion(channels, method=self.fusion_dict[self.modality_id])

        # """ Pixel-wise Channel Attention """
        # self.channel_gate = nn.Sequential(
        #     C.conv2d_1x1(channels, channels // reduction),
        #     nn.ReLU(inplace=True),
        #     C.conv2d_1x1(channels // reduction, channels),
        #     nn.Sigmoid()
        # )

        """ K-Excitations """
        self.kexc_conv = nn.Sequential(
            C.conv1d_1x1(self.h_qry * self.w_qry, num_exc),
            nn.Sigmoid()
        )
        self.adp_pool = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(self.h_qry, self.w_qry),
                stride=(self.h_qry, self.w_qry)) if ptype == 'average' else\
            nn.MaxPool2d(kernel_size=(self.h_qry, self.w_qry),
                stride=(self.h_qry, self.w_qry)) if ptype == 'maximum' else\
            nn.LPPool2d(norm_type=2, kernel_size=self.h_qry)
            for ptype in poolings
        ])

    def forward(self, x_img, x_qry):
        """
        :param x_img: image-level feature (after co-attention)
            - size : [bz, C=1024, H_i, W_i]
        :param x_qry: query-image-level feature (after co-attention)
            - size : [bz, C=1024, H_q=8, W_q=8]
        """
        bz, c_img, h_img, w_img = x_img.size()
        bz, c_qry, h_qry, w_qry = x_qry.size()

        """ Resize x_img with the same size as x_qry """
        r_img = F.interpolate(x_img, size=(h_qry, w_qry), mode=self.interpolation, align_corners=False)

        # ======== Obtain Scaling Vector(Matrix) ======== #
        """ Feature Aggregation """
        x_agg = self.aggregator(x_img=r_img, x_qry=x_qry) # [bz, C=1024, H=8, W=8]

        # """ Pixel-wise Channel Attention """
        # g_att = self.channel_gate(x_agg) # scaling matrix: [bz, C=1024, H=8, W=8]

        """ From h_qry * w_qry pixels to K-Excitations (gates) """
        x_agg = x_agg.view(bz, self.channels, -1).permute(0, 2, 1) # [bz, HW=64, C=1024]
        x_agg = self.kexc_conv(x_agg) # [bz, num_K, W=C=1024]
        scale = x_agg.permute(0, 2, 1) # [bz, C=1024, N_k=num_K]

        # ======== Excite Image-Level Feature ======== #
        x_vec = None
        for pid, pool in enumerate(self.adp_pool):
            x_pool = pool(x_qry) # [bz, C=1024, 1, 1]

            x_pool = x_pool.view(bz, self.channels, 1) # [bz, C, N=1]
            x_pool = x_pool.repeat(1, 1, self.num_exc) # [bz, C, N]
            if x_vec is None:
                x_vec = x_pool # [bz, C=1024, N=num_exc]
            else:
                x_vec = x_vec + x_pool # [bz, C=1024, N=num_exc]

        """ Excite image-level feature """
        x_exc = scale * x_vec # [bz, C, N=num_exc]

        return scale, x_exc

class FeatureFusion(nn.Module):
    def __init__(self, channels, reduction=16, num_exc=3, **kwargs):
        super(FeatureFusion, self).__init__()

        self.num_states = kwargs.get('num_states', 2) # (x_img, x_qry)
        self.num_layers = kwargs.get('num_layers', 1) # for convlstm
        self.method = kwargs.get('method', 'selective_kernel')

        self.num_exc = num_exc
        self.channels = channels

        if self.method == 'selective_kernel' or self.method == 'selective_kernel_wosplit':
            if self.method == 'selective_kernel':
                self.convs = nn.ModuleList([nn.Sequential(
                    C.conv2d_3x3(in_ch=2 * channels, out_ch=channels, dilation=d),
                    C.bn2d(channels),
                    nn.ReLU(inplace=True),
                    ) for d in range(1, self.num_states+1)
                ])
            self.se_pool = nn.AdaptiveAvgPool2d(1)
            self.se_fc = nn.Sequential(
                C.fc(channels, channels // reduction),
                C.bn1d(channels // reduction),
                nn.ReLU(inplace=True),
            )
            self.fcs = nn.ModuleList([
                C.fc(channels // reduction, channels)\
                for _ in range(self.num_states)
            ])
            self.softmax = nn.Softmax(dim=1)
        elif self.method == 'gate_conv1x1':
            self.conv = nn.Sequential(
                C.conv2d_1x1(2 * channels, channels),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
                # nn.Tanh(),
            )
        elif self.method == 'gate_conv3x3':
            self.conv = nn.Sequential(
                C.conv2d_3x3(2 * channels, channels),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
        elif self.method == 'selective_gate':
            self.g_conv_1x1 = nn.Sequential(
                C.conv2d_1x1(2 * channels, channels),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.g_conv_3x3 = nn.Sequential(
                C.conv2d_3x3(2 * channels, channels),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.convrnn_fw = C.convlstm(channels, [channels // 2]*self.num_layers,
                # kernel_size=(3, 3),
                kernel_size=(1, 1),
                num_layers=self.num_layers,
            )
            self.convrnn_bw = C.convlstm(channels, [channels // 2]*self.num_layers,
                # kernel_size=(3, 3),
                kernel_size=(1, 1),
                num_layers=self.num_layers,
            )
        elif self.method == 'gate_conv1x1_reduction':
            self.redn_conv_img = nn.Sequential(
                C.conv2d_1x1(channels, channels // reduction),
                nn.ReLU(inplace=True),
            )
            self.redn_conv_qry = nn.Sequential(
                C.conv2d_1x1(channels, channels // reduction),
                nn.ReLU(inplace=True),
            )
            self.conv = nn.Sequential(
                C.conv2d_1x1(2 * channels // reduction, channels),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
        elif self.method == 'convlstm':
            self.convrnn_fw = C.convlstm(channels, [channels // 2],
                kernel_size=(3, 3),
                num_layers=1,
            )
            self.convrnn_bw = C.convlstm(channels, [channels // 2],
                kernel_size=(3, 3),
                num_layers=1,
            )
        elif self.method == '3Dconv':
            self.conv = nn.Sequential(
                C.conv2d_1x1(2 * channels, channels * num_exc),
                nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.convrnn_fw = C.convlstm(channels, [channels // 2],
                kernel_size=(1, 1),
                num_layers=1,
            )
            self.convrnn_bw = C.convlstm(channels, [channels // 2],
                kernel_size=(1, 1),
                num_layers=1,
            )

    def flip(self, x, axis=0):
        """
        :param x.size(): [bz, S=timestamps=W, C]
        """
        device = x.device
        length = x.size(axis)

        ## functional testing snippet code
        # length = 5
        # x = torch.arange(length)
        # x = x.view(1, length, 1).to(device)
        # x = x.repeat(1, 1, 10)

        inv_idx = torch.LongTensor([i for i in range(length-1, -1, -1)]).to(device)
        inv_out = torch.index_select(x, axis, inv_idx)

        return inv_out

    def forward(self, x_img, x_qry):
        """
        :param x_img: resized image-level feature (after co-attention)
            - size : [bz, C=1024, H_i, W_i]
        :param x_qry: query-image-level feature (after co-attention)
            - size : [bz, C=1024, H_q=8, W_q=8]
        """
        bz, c_img, h_img, w_img = x_img.size()
        bz, c_qry, h_qry, w_qry = x_qry.size()

        if self.method == 'selective_kernel' or self.method == 'selective_kernel_wosplit':
            if self.method == 'selective_kernel':
                x_img_qry = torch.cat((x_img, x_qry), dim=1) # [bz, C=2048, H=8, W]

                u_feat_mat = [self.convs[ns](x_img_qry) for ns in range(self.num_states)]
                u_feat_mat = torch.stack(u_feat_mat).permute(1, 0, 2, 3, 4) # [bz, N_s=num_states, C=1024, H, W]
            elif self.method == 'selective_kernel_wosplit':
                u_feat_mat = torch.stack((x_img, x_qry)).permute(1, 0, 2, 3, 4) # [bz, N_s=num_states, C=1024, H, W]
            u_feat = torch.sum(u_feat_mat, dim=1) # [bz, C=1024, H, W]
            s_feat = self.se_pool(u_feat).view(bz, self.channels) # [bz, C=1024]
            z_feat = self.se_fc(s_feat) # [bz, C'=C / reduction]

            vector = [self.fcs[ns](z_feat) for ns in range(self.num_states)]
            vector = torch.stack(vector).permute(1, 0, 2) # [bz, N_s=num_states, C=1024]
            vector = self.softmax(vector) # [bz, N_s=num_states, C=1024]
            vector = vector.view(bz, self.num_states, self.channels, 1, 1) # [bz, N_s, C=1024, H=1, W=1]

            # Get aggregated feature matrix
            v_feat = (u_feat_mat * vector.expand_as(u_feat_mat)).sum(dim=1) # [bz, C=1024, H, W]

        elif self.method == 'gate_conv1x1' or self.method == 'gate_conv3x3':
            x_img_qry = torch.cat((x_img, x_qry), dim=1) # [bz, C=2048, H, W]

            g_feat = self.conv(x_img_qry) # [bz, C=1024, H, W]

            v_feat = x_img + x_img * g_feat
        elif self.method == '3Dconv':
            x_img_qry = torch.cat((x_img, x_qry), dim=1) # [bz, C=2048, H, W]

            gates = self.conv(x_img_qry) # [bz, C=K * 1024, H, W]

            """ chunk across channel dimension """
            g_cube = gates.chunk(self.num_exc, 1) # list of gates
            v_feat = [x_img + x_img * g_feat for g_feat in g_cube]
            x_feats = torch.stack(v_feat) # [K, bz, C=1024, H, W]

            # states first
            inv_feats = self.flip(x_feats, axis=0) # Inversed States: [NS=K, bz, C=1024, H, W]

            out_fw, _ = self.convrnn_fw(x_feats)
            out_bw, _ = self.convrnn_bw(inv_feats)

            out_bw = self.flip(out_bw, axis=0)

            out_fw = out_fw[-1] # final state: [bz, C=512, H, W]
            out_bw = out_bw[-1] # final state: [bz, C=512, H, W]

            # Get aggregated feature matrix
            v_feat = torch.cat((out_fw, out_bw), dim=1) # [bz, C=1024, H, W]

        elif self.method == 'gate_conv1x1_reduction':
            x_img_redn = self.redn_conv_img(x_img)
            x_qry_redn = self.redn_conv_qry(x_qry)
            x_img_qry = torch.cat((x_img_redn, x_qry_redn), dim=1) # [bz, C=2 * 1024 // 16, H, W]

            g_feat = self.conv(x_img_qry) # [bz, C=1024, H, W]

            v_feat = x_img + x_img * g_feat
        elif self.method == 'selective_gate':
            x_img_qry = torch.cat((x_img, x_qry), dim=1) # [bz, C=2048, H, W]

            g_feat_1x1 = self.g_conv_1x1(x_img_qry) # [bz, C=1024, H, W]
            v_feat_1x1 = x_img + x_img * g_feat_1x1 # [bz, C=1024, H, W]

            g_feat_3x3 = self.g_conv_3x3(x_img_qry) # [bz, C=1024, H, W]
            v_feat_3x3 = x_img + x_img * g_feat_3x3 # [bz, C=1024, H, W]

            x_feats = torch.stack((v_feat_1x1, v_feat_3x3)) # [NS=2, bz, C=2048, H, W]
            # states first
            inv_feats = self.flip(x_feats, axis=0) # Inversed States: [NS=2, bz, C=2048, H, W]

            out_fw, _ = self.convrnn_fw(x_feats)
            out_bw, _ = self.convrnn_bw(inv_feats)

            out_bw = self.flip(out_bw, axis=0)

            out_fw = out_fw[-1] # final state: [bz, C=512, H, W]
            out_bw = out_bw[-1] # final state: [bz, C=512, H, W]

            # Get aggregated feature matrix
            v_feat = torch.cat((out_fw, out_bw), dim=1) # [bz, C=1024, H, W]

        elif self.method == 'convlstm':
            x_img_qry = torch.stack((x_img, x_qry)) # [NS=2, bz, C=2048, H, W]

            # states first
            inv_img_qry = self.flip(x_img_qry, axis=0) # Inversed States: [NS=2, bz, C=2048, H, W]

            out_fw, _ = self.convrnn_fw(x_img_qry)
            out_bw, _ = self.convrnn_bw(inv_img_qry)

            out_bw = self.flip(out_bw, axis=0)

            out_fw = out_fw[-1] # final state: [bz, C=512, H, W]
            out_bw = out_bw[-1] # final state: [bz, C=512, H, W]

            v_feat = torch.cat((out_fw, out_bw), dim=1) # [bz, C=1024, H, W]

        return v_feat

class CoRelation(nn.Module):
    def __init__(self, num_exc, **kwargs):
        super(CoRelation, self).__init__()
        self.channels = kwargs.get('channels', 1024)
        self.with_normalization = kwargs.get('with_normalization', False)
        self.with_interpolation = kwargs.get('with_interpolation', False)

        if self.with_interpolation:
            self.interpolation = kwargs.get('interpolation', 'bilinear')

        reduction = 16
        # self.props_conv = nn.Sequential(
        #     nn.Conv1d(self.channels, self.channels // reduction,
        #         kernel_size=7*7, padding=0, stride=7*7, bias=True),
        #     # C.gn(32, self.channels),
        #     # Mish(),
        # )
        # self.query_conv = nn.Sequential(
        #     nn.Conv1d(self.channels, self.channels // reduction,
        #         kernel_size=8*8, padding=0, stride=8*8, bias=True),
        #     # C.gn(32, self.channels),
        #     # Mish(),
        # )
        self.props_conv = nn.Sequential(
            C.conv1d_1x1(self.channels, self.channels // reduction),
        )
        self.query_conv = nn.Sequential(
            C.conv1d_1x1(self.channels, self.channels // reduction),
        )

        """ Channel Gate """
        # self.channel_gate = ChannelAttention(self.channels)

        # self.softmax = nn.Softmax(dim=-1)

        self.props_mlp = nn.Sequential(
            C.Flatten(),
            # nn.Linear(self.channels, self.channels // reduction),
            # nn.ReLU(),
            nn.Linear(self.channels // reduction, self.channels),
            nn.Sigmoid(),
        )
        self.query_mlp = nn.Sequential(
            C.Flatten(),
            # nn.Linear(self.channels, self.channels // reduction),
            # nn.ReLU(),
            nn.Linear(self.channels // reduction, self.channels),
            nn.Sigmoid(),
        )

    def forward(self, x_props, x_query):
        """
        :param x_props: proposal-level feature (after co-attention + RoIAlign)
            - size: [bz * num_props, C=1024, H_p=7, W_p=7]
            - num_props:
                - training: 128
                - testing: 300
        :param x_query: query-image-level feature (after co-attention)
            - size : [bz, C=1024, H_q=8, W_q=8]
        """
        bp, c_props, h_props, w_props = x_props.size()
        bz, c_query, h_query, w_query = x_query.size()

        num_props = int(bp // bz)

        """ Resize x_query with the same size as x_props """
        if self.with_interpolation:
            r_query = F.interpolate(x_query, size=(h_props, w_props), mode=self.interpolation, align_corners=False)
        else:
            r_query = x_query

        """ For each proposal having query feature """
        r_query = r_query.unsqueeze(1).repeat(1, num_props, 1, 1, 1) # [bz, num_props, C=1024, H=7, W=7]
        # r_query = r_query.view(-1, c_props, h_props, w_props) # [bz * num_props, C=1024, H=7, W=7]

        p_feat = x_props.view(bp, c_props, -1) # Proposal: [bz * num_props, C=1024, HW=64]
        q_feat = r_query.view(bp, c_props, -1) # Query: [bz * num_props, C=1024, HW=64]

        # ======================================================================================= # 
        p_feat = self.props_conv(p_feat) # [bz * num_props ,C=1024, 1]
        q_feat = self.query_conv(q_feat) # [bz * num_props ,C=1024, 1]

        # p_feat = self.props_pool(p_feat) # [bz * num_props ,C=1024, 1]
        # q_feat = self.query_pool(q_feat) # [bz * num_props, C=1024, 1]
        # print('After: ', p_feat.size(), q_feat.size())

        # if self.with_normalization:
        #     p_feat = F.normalize(p_feat, p=2, dim=2) # L2 norm along spatial dim
        #     q_feat = F.normalize(q_feat, p=2, dim=2) # L2 norm along spatial dim

        """ Similarity """
        similarity = torch.matmul(p_feat, q_feat.permute(0, 2, 1).contiguous()) # [bz * num_props, C^p, C^q]
        # ch_relation = torch.diagonal(similarity, offset=0, dim1=-2, dim2=-1) # [bz * num_props, C=1024]

        p_relation = similarity.mean(dim=2) # [bz * num_props, C^p]
        q_relation = similarity.mean(dim=1) # [bz * num_props, C^q]
        # p_relation = similarity.max(dim=2)[0] # [bz * num_props, C^p]
        # q_relation = similarity.max(dim=1)[0] # [bz * num_props, C^q]

        """ Channel Gate """
        # c_att = self.softmax(ch_relation).view(bp, c_props, 1, 1) # [bz * num_props, C=1024, 1, 1]

        # c_att = self.mlp(ch_relation).view(bp, c_props, 1, 1) # [bz * num_props, C=1024, 1, 1]

        p_att = self.props_mlp(p_relation).view(bp, c_props, 1, 1) # [bz * num_props, C=1024, 1, 1]
        q_att = self.query_mlp(q_relation).view(bp, c_props, 1, 1) # [bz * num_props, C=1024, 1, 1]

        # ======================================================================================= # 
        # original
        # c_att = self.channel_gate(x_agg) # [bz * num_props, C, 1, 1]

        """ Co-Excitation """
        # Excite proposal-level features
        # f_props = x_props * c_att.expand_as(x_props) # [bz * num_props, C=1024, H=7, W=7]
        # f_props = x_props * q_att.expand_as(x_props) # [bz * num_props, C=1024, H=7, W=7]
        f_props = x_props * p_att.expand_as(x_props) # [bz * num_props, C=1024, H=7, W=7]

        # Excite query-imgae-level features
        x_query = x_query.unsqueeze(1).repeat(1, num_props, 1, 1, 1) # [bz, num_props, C=1024, H=8, W=8]
        x_query = x_query.view(-1, c_query, h_query, w_query) # [bz * num_props, C=1024, H=8, W=8]
        # f_query = x_query * c_att.expand_as(x_query) # [bz * num_props, C=1024, H=8, W=8]
        # f_query = x_query * p_att.expand_as(x_query) # [bz * num_props, C=1024, H=8, W=8]
        f_query = x_query * q_att.expand_as(x_query) # [bz * num_props, C=1024, H=8, W=8]

        return f_props, f_query, p_att

class SENet(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SENet, self).__init__()

        self.se_props = ChannelAttention(channels, reduction)
        self.se_query = ChannelAttention(channels, reduction)

    def forward(self, x_props, x_query):

        c_att_props = self.se_props(x_props) # [bp, C, 1, 1]
        c_att_query = self.se_query(x_query) # [bs, C, 1, 1]

        f_props = x_props * c_att_props.expand_as(x_props) # [bp, C=1024, H=7, W=7]
        f_query = x_query * c_att_query.expand_as(x_query) # [bs, C=1024, H=8, W=8]

        return f_props, f_query

class SKBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SKBlock, self).__init__()

        groups = 8
        # self.n_state = 2
        kernels = [1, 3]
        self.n_state = len(kernels)

        """ Selective Kernel Operations (SK) """
        self.convs = nn.ModuleList([nn.Sequential(
            # nn.Conv2d(channels, channels, kernel_size=3+ns*2, stride=1, padding=1+ns, groups=groups),
            # nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1+ns, groups=groups,
            #     dilation=1+ns),
            nn.Conv2d(channels, channels, kernel_size=kernels[ns], stride=1, padding=kernels[ns]//2, groups=groups),
            nn.ReLU(inplace=True),
            ) for ns in range(self.n_state)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels // reduction)
        self.sk = nn.Linear(channels // reduction, channels * self.n_state)
        self.softmax = nn.Softmax(dim=1)

        self.reset_params()

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # if m.bias is not None:
            #     fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            #     bound = 1 / math.sqrt(fan_in)
            #     nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: feature map
            - size: [bs, C, H, W]
        """

        n_state = self.n_state

        bs, C, H, W = x.size()

        f = [self.convs[ns](x) for ns in range(n_state)]
        f = torch.stack(f, dim=1) # [bs, ns, C, H, W]

        u = torch.sum(f, dim=1) # [bs, C, H, W]
        s = self.gap(u).view(bs, C) # [bs, C]
        z = self.fc(s) # [bs, C']
        a = self.sk(z) # [bs, C*ns]
        a = a.view(bs, n_state, C) # [bs, ns, C]

        a = self.softmax(a).view(bs, n_state, C, 1, 1) # [bs, ns, C, H=1, W=1]

        v = f * f.expand_as(f) # [bs, ns, C, H, W]
        v = v.sum(dim=1) # [bs, C, H, W]

        return v

class SKNet(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SKNet, self).__init__()

        self.sk_props = SKBlock(channels, reduction)
        self.sk_query = SKBlock(channels, reduction)

    def forward(self, x_props, x_query):

        f_props = self.sk_props(x_props) # [bp, C=1024, H=7, W=7]
        f_query = self.sk_query(x_query) # [bs, C=1024, H=8, W=8]

        return f_props, f_query
