from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_sys_transformer_sk_dilat import _fasterRCNN

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import copy
from lib.ops.utils import printer, color

import model.modules.cells as C

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

TERMINAL_ENVCOLS = list(map(int, os.popen('stty size', 'r').read().split()))[1]

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class RCNNBackbone(nn.Module):
    def __init__(self, cfg, backbone, **kwargs):
        super(RCNNBackbone, self).__init__()
        self.backbone = backbone

        self.channels = kwargs.get('channels', 2048)

        self.with_contextual_relation = kwargs.get('with_contextual_relation', False)
        self.rnn_layers = kwargs.get('rnn_layers', 1)
        self.reduction = kwargs.get('reduction', 16)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.rnn_method = kwargs.get('rnn_method', 'GRU')

        '''
        # Build resnet.
        self.RCNN_base = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3
        )
        '''

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        # Fix blocks
        for p in self.stem[0].parameters(): p.requires_grad=False
        for p in self.stem[1].parameters(): p.requires_grad=False

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        '''
        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4) # cfg.RESNET.FIXED_BLOCKS=2
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.layer3.parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.layer2.parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.layer1.parameters(): p.requires_grad=False
        '''

        if self.with_contextual_relation:
            self.rnn = nn.GRU(
                input_size=self.channels,
                hidden_size=self.channels // self.reduction,
                num_layers=self.rnn_layers,
                batch_first=True, bidirectional=self.bidirectional,
            ) if self.rnn_method == 'GRU' else nn.LSTM(\
                input_size=self.channels,
                hidden_size=self.channels // self.reduction,
                num_layers=self.rnn_layers,
                batch_first=True, bidirectional=self.bidirectional,
            )
            self.trans1 = nn.Sequential(
                C.conv2d_1x1(256, self.channels),
                # nn.ReLU(inplace=True),
                # Mish(),
                # Swish(),
            )
            self.trans2 = nn.Sequential(
                C.conv2d_1x1(512, self.channels),
                # nn.ReLU(inplace=True),
                # Mish(),
                # Swish(),
            )
            self.trans3 = nn.Sequential(
                C.conv2d_1x1(1024, self.channels),
                # nn.ReLU(inplace=True),
                # Mish(),
                # Swish(),
            )
            self.gap1 = nn.AdaptiveAvgPool2d(1)
            self.gap2 = nn.AdaptiveAvgPool2d(1)
            self.gap3 = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                C.fc(2 * self.channels // self.reduction if self.bidirectional\
                   else self.channels // self.reduction, self.channels),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.Sigmoid(),
            )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        x0 = self.stem(x) # [bz, 64, H0, W0]

        x1 = self.layer1(x0) # [bz, 256, H1=H0, W1=W0]
        x2 = self.layer2(x1) # [bz, 512, H2, W2]
        x3 = self.layer3(x2) # [bz, 1024, H3, W3]

        out = x3

        if self.with_contextual_relation:
            """ rnn low-high contextual relation """
            x1 = self.trans1(x1) # [bz, 2048, H1=H0, W1=W0]
            x2 = self.trans2(x2) # [bz, 2048, H2, W2]
            x3 = self.trans3(x3) # [bz, 2048, H3, W3]

            x1 = self.gap1(x1).view(batch_size, self.channels) # [bz, 2048]
            x2 = self.gap2(x2).view(batch_size, self.channels) # [bz, 2048]
            x3 = self.gap3(x3).view(batch_size, self.channels) # [bz, 2048]

            x_seq = torch.stack((x1, x2, x3)).permute(1, 0, 2) # [bz, S=3, 2048]

            hidden_tuple = (2 * self.rnn_layers if self.bidirectional else self.rnn_layers,
                            batch_size, self.channels // self.reduction)
            h0 = torch.zeros(hidden_tuple).to(device)
            if self.rnn_method == 'LSTM':
                c0 = torch.zeros(hidden_tuple).to(device)
            self.rnn.flatten_parameters()
            if self.rnn_method == 'GRU':
                rnn_out, rnn_hn = self.rnn(x_seq, h0) # [bz, S=2, C=2048]
            elif self.rnn_method == 'LSTM':
                rnn_out, rnn_hn = self.rnn(x_seq, (h0, c0)) # [bz, S=2, C=2048/16]
            scale = self.fc(rnn_out[:, -1, :])
            scale = scale.view(batch_size, self.channels)#, 1, 1)
            # out = x3 * scale.expand_as(x3) # [bz, 2048, H3, W3]
            return out, scale
        else:
            return out, None

class resnet(_fasterRCNN):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, num_K=3):
        if num_layers==50:
            self.model_path = '../data/pretrain_imagenet_resnet50/model_best.pth.tar'
        elif num_layers==101:
            self.model_path = '../data/pretrain_imagenet_resnet101/model_best.pth.tar'

        self.with_reduce_classes = True
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.num_layers = num_layers

        _fasterRCNN.__init__(self, classes, class_agnostic, num_K)

    def _init_modules(self):
        if self.num_layers==50:
            resnet = resnet50()
        else:
            resnet = resnet101()

        if self.pretrained == True:
            if self.with_reduce_classes:
                printer('Loading pretrained weights: ', prnt_info='{}'.format(self.model_path))
                state_dict = torch.load(self.model_path)
                state_dict = state_dict['state_dict']

                state_dict_v2 = copy.deepcopy(state_dict)

                for key in state_dict:
                    pre, post = key.split('module.')
                    state_dict_v2[post] = state_dict_v2.pop(key)

                resnet.load_state_dict(state_dict_v2)
            else:
                printer('Loading pretrained weights: ', prnt_info='{}'.format(model_urls['resnet50']))
                resnet.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        # Build resnet.
        '''
        self.RCNN_base = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3
        )
        # Fix blocks
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        '''
        self.RCNN_base = RCNNBackbone(cfg, backbone=resnet)

        self.RCNN_top = nn.Sequential(resnet.layer4)


        self.RCNN_cls_score = nn.Sequential(
                nn.Linear(2048*2, 8),
                nn.Linear(8, 2)
        )

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

        # for num_layer,child in enumerate(self.RCNN_base.children()):
        #   for param in child.parameters():
        #     param.requires_grad = False

        # self.model_printer()

    def model_printer(self):
        print('{sep}'.format(sep='.' * TERMINAL_ENVCOLS))
        for i, m in enumerate(self.modules()):
            class_name = str(m.__class__).split(".")[-1].split("'")[0]
            if class_name == 'resnet':
                print(m)
        # print('{sep}'.format(sep='.' * TERMINAL_ENVCOLS))

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode

            # self.RCNN_base.eval()
            # self.RCNN_base[5].train()
            # self.RCNN_base[6].train()

            self.RCNN_base.stem.eval()
            '''
            self.RCNN_base.layer1.eval()
            '''
            # self.RCNN_base.layer2.train()
            # self.RCNN_base.layer3.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        """
        : input of pool5:
            - if pooled_feat(proposals).size: [bz*num_props, 1024, 7, 7]
                - output of self.RCNN_top(pool5).size: [bz * num_props, 2048, 4, 4]
            - if act_aim(query).size: [bz, 1024, 8, 8]
                - output of self.RCNN_top(pool5).size: [bz, 2048, 4, 4]
        """
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
