
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import pdb
import cv2
import time
import torch
import pprint
import pickle
import datetime
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet_coatt_transformer_sk import resnet

# import inspect
from terminaltables import *
from lib.utilities import Bar
from lib.ops.utils import mkdir, printer, color, AverageMeter

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def save_weight(weight, time, seen):
  time = np.where(time==0, 1, time)
  weight = weight/time[:,np.newaxis]
  result_map = np.zeros((len(weight), len(weight)))
  for i in range(len(weight)):
    for j in range(len(weight)):
      v1 = weight[i]
      v2 = weight[j]
      # v1_ = np.linalg.norm(v1)
      # v2_ = np.linalg.norm(v2)
      # v12 = np.sum(v1*v2)
      # print(v12)
      # print(v1_)
      # print(v2_)
      distance = np.linalg.norm(v1-v2)
      if np.sum(v1*v2)== 0 :
        result_map[i][j] = 0
      else:
        result_map[i][j] = distance

  df = pd.DataFrame (result_map)

  ## save to xlsx file

  filepath = 'similarity_%d.xlsx'%(seen)

  df.to_excel(filepath, index=False)

  weight = weight*255


  cv2.imwrite('./weight_%d.png'%(seen), weight)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--s', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--p', dest='checkpoint',
                        help='checkpoint to load network',
                        default=1663, type=int)
    parser.add_argument('--vis', dest='visualization',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--seen', dest='seen',
                        help='Reserved: 1 training, 2 testing, 3 both', default=2, type=int)
    parser.add_argument('--a', dest='average', help='average the top_k candidate samples', default=1, type=int)
    parser.add_argument('--g', dest='group',
                        help='which group want to training/testing',
                        default=0, type=int)

    # debug mode
    parser.add_argument('--debug', dest='debug',
                        help='debug mode',
                        action='store_true')

    # version
    parser.add_argument('--version', dest='version',
                        help='model version to store different checkpiont',
                        default='1.0.0', type=str)
    # testing mode
    parser.add_argument('--with_cache_file', dest='with_cache_file',
                        help='whether to load pre-saved cached detection file (bbox)',
                        action='store_true')
    # specified checkpoint
    parser.add_argument('--specify-checkpoint', dest='specify_checkpoint',
                        help='specified checkpiont',
                        default=None, type=str)

    # num_K
    parser.add_argument('--num_k_excitation', dest='num_k_excitation',
                        help='number of k excitations',
                        default=3, type=int)

    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    """
    - Load pre-saved cached detection file
        - $ python test_net.py --dataset coco --net res50 --s 1 --checkepoch 10\
                --p 13311 --cuda --g 1 --a 4 --with_cache_file
    - Else: Use trained models by yourself
        - $ python test_net.py --dataset coco --net res50 --s 1 --checkepoch 10\
                --p 13311 --cuda --g 1 --a 4
    """
    TERMINAL_ENVROWS = list(map(int, os.popen('stty size', 'r').read().split()))[0]
    TERMINAL_ENVCOLS = list(map(int, os.popen('stty size', 'r').read().split()))[1]

    args = parse_args()

    print('{sep}\n{title}\n{sep}'.format(
        sep='='*TERMINAL_ENVCOLS,
        title='\t ◆ {info}: Dateset: {dt}, CheckEpoch: {ckep}, Group: {gid}'.format(
            info=color('Information', 'green'),
            dt=args.dataset.capitalize(), ckep=args.checkepoch, gid=args.group)
    ))

    printer('Called with args:')
    # print(args)
    args_dict = vars(args)
    title = [['KEY', 'VALUE']]
    args_info = [[k, args_dict[k]] for k in sorted(list(vars(args).keys()))]
    table = DoubleTable(title + args_info, ' Arguments ')
    print(table.table)

    if torch.cuda.is_available() and not args.cuda:
        printer("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    printer('Using config:')
    pprint.pprint(cfg, indent=4)

    # with open('experiment.info', 'w') as f:
    #     f.write('Session-{sess}_Epoch-{epo}_Version-{ver}'.format(
    #         sess=args.checksession, epo=args.checkepoch, ver=args.version))

    # Load dataset
    cfg.TRAIN.USE_FLIPPED = False
    imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu = combined_roidb(\
            args.imdbval_name, False, seen=args.seen)
    imdb_vu.competition_mode(on=True)
    dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu,\
            query_vu, 1, imdb_vu.num_classes, training=False, seen=args.seen)

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb_vu.classes, pretrained=False,
                class_agnostic=args.class_agnostic, num_K=args.num_k_excitation)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_vu.classes, 101, pretrained=False,
                class_agnostic=args.class_agnostic, num_K=args.num_k_excitation)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=False,
                class_agnostic=args.class_agnostic, num_K=args.num_k_excitation)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb_vu.classes, 152, pretrained=False,
                class_agnostic=args.class_agnostic, num_K=args.num_k_excitation)
    else:
        printer("network is not defined")
        pdb.set_trace()
    fasterRCNN.create_architecture()

    # Load checkpoint
    # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    # input_dir = os.path.join(args.load_dir, args.net, args.dataset)

    if args.specify_checkpoint:
        load_name = args.specify_checkpoint
        printer(bold_info='{tag}'.format(tag=color('Specified', 'yellow')),
                prnt_info=' checkpoint : {_file}'.format(_file=load_name))
    else:
        input_dir = os.path.join(args.load_dir, args.net, args.dataset, args.version)
        printer('Model path: ', prnt_info=input_dir)
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)

        # load_name = os.path.join(input_dir,
        #     'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        load_name = os.path.join(input_dir,
            '{dataset}_{backbone}_{framework}_session-{session}_epoch-{epoch}_step-{step}.pth'.format(
                dataset=args.dataset, backbone=args.net, framework='fasterRCNN',
                session=args.checksession, epoch=args.checkepoch, step=args.checkpoint
        ))
        printer("load checkpoint {}".format(load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    # initilize the tensor holder here.
    printer('load model successfully!')
    im_data = torch.FloatTensor(1)
    query   = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
        im_data = im_data.cuda()
        query = query.cuda()
        im_info = im_info.cuda()
        catgory = catgory.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    query = Variable(query)
    im_info = Variable(im_info)
    catgory = Variable(catgory)
    gt_boxes = Variable(gt_boxes)

    # record time
    tst_start_time = time.time()

    # visiualization
    vis = args.visualization
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0
    max_per_image = 100

    # create output Directory
    output_dir_vu = get_output_dir(imdb_vu, 'faster_rcnn_unseen')

    fasterRCNN.eval()
    for avg in range(args.average):
        dataset_vu.query_position = avg
        dataloader_vu = torch.utils.data.DataLoader(dataset_vu, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)

        data_iter_vu = iter(dataloader_vu)
    
        det_idx_rst = ratio_index_vu[0] if not args.debug else ratio_index_vu[0][:10]

        # total quantity of testing images, each images include multiple detect class
        num_images_vu = len(imdb_vu.image_index)
        # num_detect = len(ratio_index_vu[0])
        num_detect = len(det_idx_rst)

        all_boxes = [[[] for _ in xrange(num_images_vu)]
                    for _ in xrange(imdb_vu.num_classes)]

        _t = {'im_detect': time.time(), 'misc': time.time()}
        if args.group != 0:
            det_file = os.path.join(output_dir_vu, 'sess%d_g%d_seen%d_%d.pkl'%(args.checksession, args.group, args.seen, avg))
        else:
            det_file = os.path.join(output_dir_vu, 'sess%d_seen%d_%d.pkl'%(args.checksession, args.seen, avg))
        printer(bold_info='{state}'.format(state=color('Loaded', 'blue')\
                    if args.with_cache_file else color('Unloaded', 'yellow')),\
                prnt_info=' cached file: {det_file}'.format(det_file=det_file)\
                    if args.with_cache_file else ' cached file')
        print('{sep}'.format(sep='=' * TERMINAL_ENVCOLS))

        # if os.path.exists(det_file):
        if os.path.exists(det_file) and args.with_cache_file:
            with open(det_file, 'rb') as fid:
                all_boxes = pickle.load(fid)
        else:
            # iter_time = AverageMeter()
            # end = time.time()
            bar = Bar('[ {a_title}:{cnt:2d} ]'.format(
                a_title=color('AvgIdx', 'blue'), cnt=avg), max=num_detect)
            # for i, index in enumerate(ratio_index_vu[0]):
            with torch.no_grad():
                for i, index in enumerate(det_idx_rst):
                    with torch.no_grad():
                        data = next(data_iter_vu)
                        im_data.resize_(data[0].size()).copy_(data[0])
                        query.resize_(data[1].size()).copy_(data[1])
                        im_info.resize_(data[2].size()).copy_(data[2])
                        gt_boxes.resize_(data[3].size()).copy_(data[3])
                        catgory.resize_(data[4].size()).copy_(data[4])

                    # Run Testing
                    det_tic = time.time()
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, _, RCNN_loss_bbox, \
                    rois_label, weight = fasterRCNN(im_data, query, im_info, gt_boxes, catgory)

                    """ size
                    :var rois: [bz=1, num_props=300, 5]
                    :var cls_prob: [bz=1, 300, 1]
                    :var bbox_pred: [bz=1, 300, 4]
                    :var weight: [bz=1, C=1024, 1, 1]
                    """

                    scores = cls_prob.data
                    boxes = rois.data[:, :, 1:5]

                    # Apply bounding-box regression 
                    """
                    - cfg.TEST.BBOX_REG: True
                    - cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED: True
                    - args.class_agnostic: True
                    """
                    if cfg.TEST.BBOX_REG:
                        # Apply bounding-box regression deltas
                        box_deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                            if args.class_agnostic:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(\
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4)
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(\
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    # Resize to original ratio
                    pred_boxes /= data[2][0][2].item()

                    # Remove batch_size dimension
                    scores = scores.squeeze()
                    pred_boxes = pred_boxes.squeeze()

                    # Record time
                    det_toc = time.time()
                    detect_time = det_toc - det_tic
                    misc_tic = time.time()

                    # Post processing
                    inds = torch.nonzero(scores>thresh).view(-1)
                    if inds.numel() > 0:
                        # remove useless indices
                        cls_scores = scores[inds]
                        cls_boxes = pred_boxes[inds, :]
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                        # rearrange order
                        _, order = torch.sort(cls_scores, 0, True)
                        cls_dets = cls_dets[order]

                        # NMS
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        all_boxes[catgory][index] = cls_dets.cpu().numpy()

                    # Limit to max_per_image detections *over all classes*
                    if max_per_image > 0:
                        try:
                            image_scores = all_boxes[catgory][index][:,-1]
                            if len(image_scores) > max_per_image:
                                image_thresh = np.sort(image_scores)[-max_per_image]

                                keep = np.where(all_boxes[catgory][index][:,-1] >= image_thresh)[0]
                                all_boxes[catgory][index] = all_boxes[catgory][index][keep, :]
                        except:
                            pass

                    # measure elapsed time
                    misc_toc = time.time()
                    nms_time = misc_toc - misc_tic

                    # iter_time.update(time.time() - end)
                    # end = time.time()

                    # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                    #     .format(i + 1, num_detect, detect_time, nms_time))
                    # sys.stdout.flush()

                    bar.suffix = \
                        '({cnt:4d}/{size:4d})'\
                        ' | Total: {total:} | ETA: {eta:}'\
                        ' | Time[det]: {detect_time:.3f}s | Time[nms]: {nms_time:.3f}s'\
                        .format(
                            cnt=i + 1, size=num_detect,
                            total=bar.elapsed_td, eta=bar.eta_td,
                            detect_time=detect_time, nms_time=nms_time
                        )
                    bar.next()

                    # save test image
                    if vis and i%1==0:
                        im2show = cv2.imread(dataset_vu._roidb[dataset_vu.ratio_index[i]]['image'])
                        im2show = vis_detections(im2show, 'shot', cls_dets.cpu().numpy(), 0.8)

                        o_query = data[1][0].permute(1, 2,0).contiguous().cpu().numpy()
                        o_query *= [0.229, 0.224, 0.225]
                        o_query += [0.485, 0.456, 0.406]
                        o_query *= 255
                        o_query = o_query[:,:,::-1]

                        (h,w,c) = im2show.shape
                        o_query = cv2.resize(o_query, (h, h),interpolation=cv2.INTER_LINEAR)
                        im2show = np.concatenate((im2show, o_query), axis=1)

                        cv2.imwrite('./test_img/%d_d.png'%(i), im2show)
            bar.finish()

            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        printer('Evaluating detections')
        imdb_vu.evaluate_detections(all_boxes, output_dir_vu, save_results=False)

        tst_elap_time = time.time() - tst_start_time
        tst_h, tst_m, tst_s = str(datetime.timedelta(seconds=tst_elap_time)).split(":")
        printer("Elapsed time: ", prnt_info='{h}h:{m}m:{s:.3f}s'.format(h=tst_h, m=tst_m, s=float(tst_s)))
