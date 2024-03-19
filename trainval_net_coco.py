# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet_coatt_transformer_sk import resnet

from terminaltables import *
from lib.utilities import Bar
from lib.ops.utils import mkdir, printer, color, AverageMeter

from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res50', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
                        # default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        # default=10, type=int)
                        default=1, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--g', dest='group', type=int,
                        help='which group to train, split coco to four group',
                        default=0)
    parser.add_argument('--seen', dest='seen',default=1, type=int)

    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=128, type=int)
    parser.add_argument('--bs_v', dest='batch_size_val',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--session', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default=True)

    # debug mode
    parser.add_argument('--debug', dest='debug',
                        help='debug mode',
                        action='store_true')

    # version
    parser.add_argument('--version', dest='version',
                        help='model version to store different checkpiont',
                        default='1.0.0', type=str)

    # num_K
    parser.add_argument('--num_k_excitation', dest='num_k_excitation',
                        help='number of k excitations',
                        default=3, type=int)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    TERMINAL_ENVROWS = list(map(int, os.popen('stty size', 'r').read().split()))[0]
    TERMINAL_ENVCOLS = list(map(int, os.popen('stty size', 'r').read().split()))[1]

    args = parse_args()
    val = False

    printer('Called with args:')
    # print(args)
    args_dict = vars(args)
    title = [['KEY', 'VALUE']]
    args_info = [[k, args_dict[k]] for k in sorted(list(vars(args).keys()))]
    table = DoubleTable(title + args_info, ' Arguments ')
    print(table.table)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    printer("Using Config:")
    pprint.pprint(cfg, indent=4)
    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        printer("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    # create dataloader
    imdb, roidb, ratio_list, ratio_index, query = combined_roidb(args.imdb_name, True, seen=args.seen)
    train_size = len(roidb)
    printer('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, query, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    # create output directory
    # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = os.path.join(args.save_dir, args.net, args.dataset, args.version)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    mkdir(output_dir)
    printer('Output target: ', prnt_info=output_dir)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    query   = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        query   = query.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        cfg.CUDA = True

    # make variable
    im_data = Variable(im_data)
    query   = Variable(query)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)


    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic,
                num_K=args.num_k_excitation)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                num_K=args.num_k_excitation)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,
                num_K=args.num_k_excitation)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic,
                num_K=args.num_k_excitation)
    else:
        printer("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
            'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        printer("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        printer("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        gpu_list = args.gpus
        if len(gpu_list) == 1 and gpu_list[0] != 0:
            gpu_list = [g for g in range(gpu_list[0])]
        fasterRCNN = nn.DataParallel(fasterRCNN, device_ids=gpu_list)
        # fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size) if not args.debug else 5

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    print('{sep}'.format(sep='-' * TERMINAL_ENVCOLS))
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)

        """ Meter """
        iter_time = AverageMeter()
        losses = AverageMeter()
        fg_counter = AverageMeter()
        bg_counter = AverageMeter()
        losses_rpn_cls = AverageMeter()
        losses_rpn_box = AverageMeter()
        losses_rcnn_cls = AverageMeter()
        losses_margin = AverageMeter()
        losses_rcnn_box = AverageMeter()

        end = time.time()
        bar = Bar('[{s_title}:{s:2d} | {e_title}:{e:2d}]'.format(
            s_title=color('Session', 'blue'), e_title=color('Epoch', 'blue'),
            s=args.session, e=epoch), max=iters_per_epoch)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.resize_(data[0].size()).copy_(data[0])
            query.resize_(data[1].size()).copy_(data[1])
            im_info.resize_(data[2].size()).copy_(data[2])
            gt_boxes.resize_(data[3].size()).copy_(data[3])
            num_boxes.resize_(data[4].size()).copy_(data[4])

            # ts = time.time()

            """
            - time(fasterRCNN) in sec: 1.088

            - for training: loss and label
                - rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, margin_loss, RCNN_loss_box, rois_label
            - for testing: bbox and prob.
                - rois, cls_prob, bbox_pred, weight (visualization)
            """
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, margin_loss, RCNN_loss_box, \
            rois_label, _ = fasterRCNN(im_data, query, im_info, gt_boxes, num_boxes)

            # t_elap = time.time() - ts
            # print('')
            # print('Model: ',str(timedelta(seconds=t_elap)))

            """ cost balance """
            cost_rpn_cls = rpn_loss_cls.mean()
            cost_rpn_box = rpn_loss_box.mean()
            cost_rcnn_cls = RCNN_loss_cls.mean()
            cost_rcnn_box = RCNN_loss_box.mean()
            cost_margin = margin_loss.mean()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

            cost = cost_rpn_cls +\
                   cost_rpn_box +\
                   cost_rcnn_cls +\
                   cost_rcnn_box +\
                   cost_margin

            # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            #      + RCNN_loss_cls.mean() + margin_loss.mean() + RCNN_loss_box.mean()
            # loss_temp += loss.item()

            losses.update(cost.item(), 1)
            losses_rpn_cls.update(cost_rpn_cls, 1)
            losses_rpn_box.update(cost_rpn_box, 1)
            losses_rcnn_cls.update(cost_rcnn_cls, 1)
            losses_rcnn_box.update(cost_rcnn_box, 1)
            losses_margin.update(cost_margin, 1)
            fg_counter.update(fg_cnt, 1)
            bg_counter.update(bg_cnt, 1)

            # backward
            optimizer.zero_grad()
            cost.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            # if step % args.disp_interval == 0:
            #     if step > 0:
            #         loss_temp /= (args.disp_interval + 1)
            #     if args.mGPUs:
            #         loss_rpn_cls  = rpn_loss_cls.mean().item()
            #         loss_rpn_box  = rpn_loss_box.mean().item()
            #         loss_rcnn_cls = RCNN_loss_cls.mean().item()
            #         loss_margin   = margin_loss.mean().item()
            #         loss_rcnn_box = RCNN_loss_bbox.mean().item()
            #         fg_cnt = torch.sum(rois_label.data.ne(0))
            #         bg_cnt = rois_label.data.numel() - fg_cnt
            #     else:
            #         loss_rpn_cls = rpn_loss_cls.item()
            #         loss_rpn_box = rpn_loss_box.item()
            #         loss_rcnn_cls = RCNN_loss_cls.item()
            #         loss_margin = margin_loss.item()
            #         loss_rcnn_box = RCNN_loss_bbox.item()
            #         fg_cnt = torch.sum(rois_label.data.ne(0))
            #         bg_cnt = rois_label.data.numel() - fg_cnt

            # measure elapsed time
            iter_time.update(time.time() - end)
            end = time.time()

            # 'iter: {it:.1f}s'
            # it=iter_time.val,
            # bar.suffix =\
            #     '({step:4d}/{size:4d})'\
            #     ' | Total: {total:} | ETA: {eta:}'\
            #     ' | LR: {rate:.2e} | Loss: {loss:.3f}'\
            #     ' | FG/BG: {fg:d}/{bg:d}'\
            #     ' | RPN[c/b]: {rpn_cls:.3f}/{rpn_box:.3f}'\
            #     ' | RCNN[c/b]: {rcnn_cls:.3f}/{rcnn_box:.3f}'\
            #     ' | Margin: {margin:.3f}'\
            #     .format(
            #         step=step, size=iters_per_epoch,
            #         total=bar.elapsed_td, eta=bar.eta_td,
            #         rate=lr,
            #         loss=losses.avg,
            #         fg=fg_counter.avg, bg=bg_counter.avg,
            #         rpn_cls=losses_rpn_cls.avg, rpn_box=losses_rpn_box.avg,
            #         rcnn_cls=losses_rcnn_cls.avg, rcnn_box=losses_rcnn_box.avg,
            #         margin=losses_margin.avg,
            #     )
            bar.next()

            if args.use_tfboard:
                # ts = time.time()
                info = {
                    'loss': losses.avg,
                    'loss_rpn_cls': losses_rpn_cls.avg,
                    'loss_rpn_box': losses_rpn_box.avg,
                    'loss_rcnn_cls': losses_rcnn_cls.avg,
                    'loss_rcnn_box': losses_rcnn_box.avg,
                    'loss_margin': losses_margin.avg,
                }
                logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

            loss_temp = 0
            start = time.time()
        bar.finish()

        # save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_name = os.path.join(output_dir,
            '{dataset}_{backbone}_{framework}_session-{session}_epoch-{epoch}_step-{step}.pth'.format(
                dataset=args.dataset, backbone=args.net, framework='fasterRCNN',
                session=args.session, epoch=epoch, step=step
        ))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('')
        printer('Model saved: {}'.format(color(save_name, 'green')))
        print('{sep}'.format(sep='=' * TERMINAL_ENVCOLS))

    if args.use_tfboard:
        logger.close()
