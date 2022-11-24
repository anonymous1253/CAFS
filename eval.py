import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset, SemiDataset_OSamp
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from mmcv.utils import Config
from mmseg.models import build_segmentor

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model_config', type=str, required=True)
parser.add_argument('--cafs-pretrained-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)



def main():
    class_name = ['background','aeroplane','bicycle','bird','boat','bottle','bus',\
                    'car','cat','chair','cow','dining table','dog','horse',\
                    'motorbike', 'person','potted plant','sheep','sofa','train','tv/monitor']
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    cudnn.enabled = True
    cudnn.benchmark = True

    if cfg['use_segformer']:
        model_cfg = Config.fromfile(args.model_config)
        model = build_segmentor(
            model_cfg.model,
            train_cfg=model_cfg.get('train_cfg'),
            test_cfg=model_cfg.get('test_cfg'))
        if rank == 0:
            logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model.load_state_dict(torch.load(args.cafs_pretrained_path))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
        
    else:
        model = DeepLabV3Plus(cfg)
        if rank == 0:
            logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model.load_state_dict(torch.load(args.cafs_pretrained_path))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
        
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    if cfg['dataset'] == 'cityscapes':
        eval_mode = 'sliding_window'
    else:
        eval_mode = 'original'
        
    mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

    if rank == 0:
        for i_num,cl_name in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{} {}] IoU: {:.2f}'.format(i_num, class_name[i_num], cl_name))
        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))


if __name__ == '__main__':
    main()
