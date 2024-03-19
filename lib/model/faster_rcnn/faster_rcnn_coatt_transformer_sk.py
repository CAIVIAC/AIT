import os
import pdb
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from datetime import timedelta
from torch.autograd import Variable

from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.roi_layers import ROIAlign, ROIPool
from model.utils.net_utils import *
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

# Custom modules
import model.modules.cells as C
import model.modules.blocks_coatt_transformer_sk as B
from model.system.Models import Transformer
from model.system.SubLayers import MultiHeadAttention

TERMINAL_ENVCOLS = list(map(int, os.popen('stty size', 'r').read().split()))[1]
# class CoAttentionModule(nn.Module):
#     def __init__(self, d_word_vec, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
#         super(CoAttentionModule, self).__init__()
#         """
#         - d_model = 512
#         - d_word_vec = 1024
#         - d_inner = 2048
#         - n_head = 8
#         - d_k, d_v = 64, 64
#         - dropout = 0.1
#         """
# 
#         self.d_model = d_model
#         self.d_word_vec = d_word_vec
# 
#         self.img_emb = nn.Sequential(
#             C.conv2d_1x1(d_word_vec, d_model, bias=True),
#         )
#         self.qry_emb = nn.Sequential(
#             C.conv2d_1x1(d_word_vec, d_model, bias=True),
#         )
# 
#         self.i2q_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.q2i_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
# 
#         self.img_trans = nn.Sequential(
#             nn.Linear(d_model, d_word_vec, bias=True),
#         )
#         self.qry_trans = nn.Sequential(
#             nn.Linear(d_model, d_word_vec, bias=True),
#         )
# 
#         # self.in_ch = inplanes
#         # self.c_hidden = None
# 
#     def forward(self, x_img, x_qry):
#         """
#         :param x_img: image_feat
#             - size: [bs, C=1024, H_i, W_i]
#         :param x_qry: query_feat
#             - size : [bs, C=1024, H_q=8, W_q=8]
# 
#         :return: non_img: image-level feature (after co-attention)
#             - size : [bs, C=1024, H_i, W_i]
#         :return: non_qry: query-image-level feature (after co-attention)
#             - size : [bs, C=1024, H_q=8, W_q=8]
#         """
# 
#         d_model, d_word_vec = self.d_model, self.d_word_vec
#         bs, c_i, h_i, w_i = x_img.size()
#         bs, c_q, h_q, w_q = x_qry.size()
# 
#         x_img = self.img_emb(x_img) # [bs, C=512, H_i, W_i]
#         x_qry = self.qry_emb(x_qry) # [bs, C=512, H=8, W=8]
# 
#         x_img = x_img.view(bs, d_model, -1).transpose(1, 2) # [bs, HW, C=512]
#         x_qry = x_qry.view(bs, d_model, -1).transpose(1, 2) # [bs, HW=64, C=512]
# 
#         enc_img, _ = self.q2i_attn(
#             q=x_img, k=x_qry, v=x_qry, mask=None) # [bs, HW, C=512]
#         enc_qry, _ = self.i2q_attn(
#             q=x_qry, k=x_img, v=x_img, mask=None) # [bs, HW=64, C=512]
# 
#         enc_img = self.img_trans(enc_img) # [bs, HW, C=1024]
#         enc_qry = self.qry_trans(enc_qry) # [bs, HW=64, C=1024]
# 
#         enc_img = enc_img.transpose(1, 2) # [bs, C=1024, HW]
#         enc_qry = enc_qry.transpose(1, 2) # [bs, C=1024, HW=64]
# 
#         non_img = enc_img.view(bs, d_word_vec, h_i, w_i) # [bs, C=1024, H_i, W_i]
#         non_qry = enc_qry.view(bs, d_word_vec, h_q, w_q) # [bs, C=1024, H=8, W=8]
#         return non_img, non_qry

class CoAttentionModule(nn.Module):
    # def __init__(self, inplanes, num_K):
    def __init__(self, inplanes):
        super(CoAttentionModule, self).__init__()
        self.sub_sample = False

        self.in_ch = inplanes
        self.c_hidden = None

        if self.c_hidden is None:
            self.c_hidden = self.in_ch // 2
            if self.c_hidden == 0:
                self.c_hidden = 1

        """ Co-Attention """
        self.coattention = B.CoAttention(
            in_ch=self.in_ch, c_hidden=self.c_hidden,
            with_residual=True,
            # normlization='softmax',
            normlization='division',
        )

    def forward(self, x_img, x_qry):
        """
        :param x_img: image_feat
            - size: [bz, C=1024, H_i, W_i]
        :param x_qry: query_feat
            - size : [bz, C=1024, H_q=8, W_q=8]

        :return: non_img: image-level feature (after co-attention)
            - size : [bz, C=1024, H_i, W_i]
        :return: non_qry: query-image-level feature (after co-attention)
            - size : [bz, C=1024, H_q=8, W_q=8]
        :return: kexc_qry_vec: weighted query-image-level feature with K exictations
            - size : [bz, C=1024, num_K]
        :return: c_att: scaling vector
            - size : [bz, C=1024, num_K]
        """

        """ Co-Attention """
        non_img, non_qry = self.coattention(x_img, x_qry)

        """
        :size non_img = exc_img:
            [bz, C=1024, H=50, W=38]
            [bz, C=1024, H=38, W=67]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=50, W=38]
            [bz, C=1024, H=38, W=54]
        """

        return non_img, non_qry

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, num_K):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.channels = self.dout_base_model

        self.num_K = num_K

        self.coattention_module = CoAttentionModule(self.channels)#, num_K=self.num_K)

        # """ Losses """
        # self.RCNN_loss_cls = 0
        # self.RCNN_loss_bbox = 0

        """ Define RPN """
        self.RCNN_rpn = _RPN(self.channels)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        """ Pooling Layer
        :param cfg.POOLING_SIZE = 7
        """
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        """ Channel Attention """
        # self.corelation = B.CoRelation(num_exc=self.num_K)
        # self.se = B.SENet(channels=self.channels)
        self.sk = B.SKNet(channels=self.channels)

        """ Transformer """
        self.transformer = Transformer(
            d_k=64,
            d_v=64,
            d_model=self.channels//2,
            d_word_vec=self.channels//2,
            d_inner=self.channels*2,
            n_position=8*8,
            # n_layers=6,
            n_layers=1,
            n_head=8,
            dropout=0.1)

        """ Loss Function """
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = cfg.TRAIN.MARGIN)

        self.model_printer()

    def model_printer(self):
        print('{sep}'.format(sep='.' * TERMINAL_ENVCOLS))
        for i, m in enumerate(self.modules()):
            class_name = str(m.__class__).split(".")[-1].split("'")[0]
            if class_name == 'resnet':
                print(m)
        print('{sep}'.format(sep='.' * TERMINAL_ENVCOLS))

    def forward(self, image, query, img_info, gt_boxes, num_boxes):
        bs = image.size(0)

        img_info = img_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        with_cxt_rel = self.RCNN_base.with_contextual_relation

        # feed image data to base model to obtain base feature map (Siames Network)
        image_feat, image_scale = self.RCNN_base(image) # [bs, C=1024, H_i, W_i]
        query_feat, query_scale = self.RCNN_base(query) # [bs, C=1024, H_q=8, W_q=8]

        # non_img, exc_img, exc_qry, c_att = self.coattention_module(image_feat, query_feat)
        # non_img, non_qry = self.coattention(x_img=image_feat, x_qry=query_feat)
        non_img, non_qry = self.coattention_module(image_feat, query_feat)
        """
        :var non_img: [bs, C=1024, H_i, W_i]
        :var non_qry: [bs, C=1024, H_q=8, W_q=8]
        :var exc_img = non_img * c_att: [bs, C=1024, H_i, W_i]
        :var exc_qry = non_qry * c_att: [bs, C=1024, H_q=8, W_q=8]
        :var c_att: [bs, C=1024, H_c=1, W_c=1] (baseline)
        :var c_att: [bs, C=1024, num_K] (revised)
        :var kexc_qry_vec: [bs, C=1024, num_K]
        """

        """ time(RCNN_rpn) in sec: 0.97 / 1.088 """
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(non_img, img_info, gt_boxes, num_boxes)

        """
        :var rois: [bs, num_props=N_p=2000, coordinate=5]
        """

        """ time(RCNN_proposal_target) in sec: 0.1 / 1.088 """
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            margin_loss = 0
            rpn_loss_bbox = 0
            score_label = None

        """ time(RCNN_roi_align) in sec: 0.000169 / 1.088 """
        rois = Variable(rois) # [bs, 128, 5]
        num_props = rois.size(1)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            props_feat = self.RCNN_roi_align(non_img, rois.view(-1, 5))
            # props_feat = self.RCNN_roi_align(exc_img, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            props_feat = self.RCNN_roi_pool(non_img, rois.view(-1, 5))
            # props_feat = self.RCNN_roi_pool(exc_img, rois.view(-1, 5))
        """
        :var props_feat: [2048=bs*128, 1024, 7, 7]
        """

        """ Transformer """
        props_feat = self.transformer(x_props=props_feat, x_query=non_qry) # [bp, C=1024, H=8, W=8]

        # """ Co-Relation """
        # props_feat, query_feat, c_att = self.corelation(x_props=props_feat, x_query=non_qry)
        """ Channel Attention """
        props_feat, query_feat = self.sk(x_props=props_feat, x_query=non_qry)
        c_att = None

        """ time(RCNN_roi_align) in sec: 0.002365 / 1.088 """
        # feed pooled features to top model
        props_feat = self._head_to_tail(props_feat) # [bs*num_props=2048, C'=2048]
        query_feat = self._head_to_tail(query_feat) # [bs, C'=2048]
        # query_feat = self._head_to_tail(exc_qry) # [bs, C'=2048]

        if with_cxt_rel:
            """ Recurrent Excitation """
            image_scale = image_scale.unsqueeze(1).repeat(1, num_props, 1) # [bs, num_props, C=2048]
            query_scale = query_scale.unsqueeze(1).repeat(1, num_props, 1) # [bs, num_props, C=2048]
            image_scale = image_scale.view(-1, image_scale.size(-1))
            query_scale = query_scale.view(-1, query_scale.size(-1))
            # props_feat = props_feat * image_scale
            # query_feat = query_feat * query_scale
            props_feat = props_feat * query_scale
            query_feat = query_feat * query_scale

        # compute bbox offset
        """
        :func RCNN_bbox_pred = nn.Linear(2048, 4)
        """
        bbox_pred = self.RCNN_bbox_pred(props_feat) # [bs*128=2048, 4]

        props_feat = props_feat.view(bs, num_props, -1) # [bs, 128, C'=2048]
        # query_feat = query_feat.view(bs, num_props, -1) # [bs, 128, C'=2048]
        query_feat = query_feat.unsqueeze(1).repeat(1, num_props, 1) # [bs, 128, C'=2048]

        # ======== Added codes (End) ======= #
        stack_feat = torch.cat((props_feat, query_feat), dim=2).view(-1, 4096)

        # compute object classification probability
        """
        :func RCNN_cls_score = nn.Sequential(
                              nn.Linear(2048*2, 8),
                              nn.Linear(8, 2)
                              )
        - time(RCNN_cls_score) in sec: 0.000081 / 1.088
        """
        score = self.RCNN_cls_score(stack_feat) # [bs*128=2048, 2]

        score_prob = F.softmax(score, 1)[:, 1] # [bs*128=2048]

        """ Losses """
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        """ time(loss) in sec: 0.000621 / 1.088 """
        if self.training:
            # classification loss
            """ rois_label: [bs*N_p=2048] """
            score_label = rois_label.view(bs, -1).float() # [bs, num_props=N_p=128]
            gt_map = torch.abs(score_label.unsqueeze(1) - score_label.unsqueeze(-1)) # [bs, N_p=128, N_p=128]

            score_prob = score_prob.view(bs, -1) # [bs, num_props=128]
            pr_map = torch.abs(score_prob.unsqueeze(1) - score_prob.unsqueeze(-1)) # [bs, N_p=128, N_p=128]
            target = -((gt_map - 1)**2) + gt_map # [bs, N_p=128, N_p=128]

            RCNN_loss_cls = F.cross_entropy(score, rois_label) # NOTE: why not : (score_prob, rois_label) ?

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)

            # RCNN_loss_cls = similarity + margin_loss

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = score_prob.view(bs, rois.size(1), -1)
        bbox_pred = bbox_pred.view(bs, rois.size(1), -1)

        """ size
        :var rois: [bs, 128, 5]
        :var cls_prob: [bs, 128, 1]
        :var bbox_pred: [bs, 128, 4]
        :var rois_label: [bs*128 = 2048]
        :var c_att: [bs, C=1024, 1, 1]
        """
        return rois, cls_prob, bbox_pred,\
               rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, margin_loss, RCNN_loss_bbox,\
               rois_label, c_att

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
