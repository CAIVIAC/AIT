import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import *
from datetime import timedelta

class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()
        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        self.ChannelGate = ChannelGate(self.in_channels) # ChannelGate in net_uilts.py
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, detect, aim):
        """
        :param detect: detect_feat
            - size: [bz, C_in=1024, H_d, W_d]
        :param aim: query_feat
            - size : [bz, C_in=1024, H_a=8, W_a=8]
        """
        batch_size, channels, height_a, width_a = aim.shape
        batch_size, channels, height_d, width_d = detect.shape

        # ==================== Find aim image similar object ==================== #
        """
        self.g, self.phi, self.theta: Conv2d_1x1: in_channels --> self.inter_channels
        """
        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous() # [bz, N_d=H_d*W_d, C=512]

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous() # [bz, N_a=H_a*W_a=64, C=512]

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # [bz, N_a=64, C=512]

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1) # [bz, C=512, N_d]

        f = torch.matmul(theta_x, phi_x) # [bz, N_a=64, N_d=N]

        N = f.size(-1)
        f_div_C = f / N # [bz, N_a=64, N_d=N]

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N # [bz, N_d=N, N_a=64]

        """ Obtain non-local feature
        - F(I) = non_det = non-local feature of Input Image
        - F(Q) = non_aim = non-local feature of Query Image
        """
        non_aim = torch.matmul(f_div_C, d_x) # [bz, N_a=64, C=512]
        non_aim = non_aim.permute(0, 2, 1).contiguous() # [bz, C=512, N_a=64]
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a) # [bz, C=512, H_a=8, W_a=8]
        non_aim = self.W(non_aim) # [bz, C_in=1024, H_a=8, W_a=8]
        non_aim = non_aim + aim # non-local residual: [bz, C_in=1024, H_a=8, W_a=8] = F(Q)

        non_det = torch.matmul(fi_div_C, a_x) # [bz, N_d, C=512]
        non_det = non_det.permute(0, 2, 1).contiguous() # [bz, C=512, N_d]
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d) # [bz, C=512, H_d, W_d]
        non_det = self.Q(non_det) # [bz, C_in=1024, H_d, W_d]
        non_det = non_det + detect # non-local residual: [bz, C_in=1024, H_d, W_d] = F(I)

        # ==================== Co-Excitation: Response in channel weight (attention) ==================== #
        c_weight = self.ChannelGate(non_aim) # [bz, C_in=1024, H_c=1, W_c=1]
        act_aim = non_aim * c_weight # [bz, C_in=1024, H_a=8, W_a=8] = bar{F(Q)}
        act_det = non_det * c_weight # [bz, C_in=1024, H_d, W_d] = bar{F(I)}

        """
        :size non_det = act_det:
            [bz, C=1024, H=50, W=38]
            [bz, C=1024, H=38, W=67]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=38, W=50]
            [bz, C=1024, H=50, W=38]
            [bz, C=1024, H=38, W=54]
        """

        # rpn_feat, act_feat, act_aim, c_weight = self.match_net(detect_feat, query_feat)
        return non_det, act_det, act_aim, c_weight

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.match_net = match_block(self.dout_base_model)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = cfg.TRAIN.MARGIN)

    def forward(self, im_data, query, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        detect_feat = self.RCNN_base(im_data) # [bz, C=1024, H_d, W_d]
        query_feat = self.RCNN_base(query) # [bz, C=1024, H_q=8, W_q=8]

        rpn_feat, act_feat, act_aim, c_weight = self.match_net(detect_feat, query_feat)
        """
        :var rpn_feat = non_det: [bz, C=1024, H_d, W_d]
        :var act_feat = act_det = non_det * c_weight: [bz, C=1024, H_d, W_d]
        :var act_aim = non_aim * c_weight: [bz, C=1024, H_a=8, W_a=8]
        :var c_weight: [bz, C_in=1024, H_c=1, W_c=1]
        """

        """ time(RCNN_rpn) in sec: 0.97 / 1.088 """
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feat, im_info, gt_boxes, num_boxes)

        """
        :var rois: [bz, num_props=N_p=2000, coordinate=5]
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
        rois = Variable(rois) # [bz, 128, 5]
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(act_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(act_feat, rois.view(-1,5))

        """
        :var pooled_feat: [2048=bz*128, 1024, 7, 7]
        """

        """ time(RCNN_roi_align) in sec: 0.002365 / 1.088 """
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat) # [bz*128=2048, C'=2048]
        query_feat  = self._head_to_tail(act_aim) # [bz, C'=2048]

        # compute bbox offset
        """
        :func RCNN_bbox_pred = nn.Linear(2048, 4)
        """
        bbox_pred = self.RCNN_bbox_pred(pooled_feat) # [bz*128=2048, 4]

        pooled_feat = pooled_feat.view(batch_size, rois.size(1), -1) # [bz, 128, C'=2048]
        query_feat = query_feat.unsqueeze(1).repeat(1,rois.size(1),1) # [bz, 128, C'=2048]

        pooled_feat = torch.cat((pooled_feat,query_feat), dim=2).view(-1, 4096)

        # compute object classification probability
        """
        :func RCNN_cls_score = nn.Sequential(
                              nn.Linear(2048*2, 8),
                              nn.Linear(8, 2)
                              )
        - time(RCNN_cls_score) in sec: 0.000081 / 1.088
        """
        score = self.RCNN_cls_score(pooled_feat) # [bz*128=2048, 2]

        score_prob = F.softmax(score, 1)[:,1] # [bz*128=2048]


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        """ time(loss) in sec: 0.000621 / 1.088 """
        if self.training:
            # classification loss
            """ rois_label: [bz*N_p=2048] """
            score_label = rois_label.view(batch_size, -1).float() # [bz, num_props=N_p=128]
            gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1)) # [bz, N_p=128, N_p=128]

            score_prob = score_prob.view(batch_size, -1) # [bz, num_props=128]
            pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1)) # [bz, N_p=128, N_p=128]
            target = -((gt_map-1)**2) + gt_map # [bz, N_p=128, N_p=128]

            RCNN_loss_cls = F.cross_entropy(score, rois_label) # NOTE: why not : (score_prob, rois_label) ?

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)

            # RCNN_loss_cls = similarity + margin_loss

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = score_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ size
        :var rois: [bz, 128, 5]
        :var cls_prob: [bz, 128, 1]
        :var bbox_pred: [bz, 128, 4]
        :var rois_label: [bz*128 = 2048]
        :var c_weight: [bz, C=1024, 1, 1]
        """
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, margin_loss, RCNN_loss_bbox, rois_label, c_weight

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
