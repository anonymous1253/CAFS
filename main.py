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
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def threshold_pseudo(pred_soft, class_threshold):
    batch_size, num_class, h, w = pred_soft.shape
    pred_mask = torch.zeros(batch_size,h,w).cuda()
    
    for c_num in range(num_class):
        pred_mask += torch.where(pred_soft[:,c_num,:,:]>=class_threshold[c_num],1,0)
    
    return pred_mask.bool()


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

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    if cfg['use_segformer']:
        model_cfg = Config.fromfile(args.model_config)
        model = build_segmentor(
            model_cfg.model,
            train_cfg=None,
            test_cfg=None)
        
        if rank == 0:
            logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
        
        class_iou = np.load(args.labeled_id_path.replace('labeled.txt', 'segformer_class_iou.npy')) 
        class_threshold = np.load(args.labeled_id_path.replace('labeled.txt', 'segformer_class_threshold.npy')) 
        
    else:
        model = DeepLabV3Plus(cfg)
        if rank == 0:
            logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
        
        class_iou = np.load(args.labeled_id_path.replace('labeled.txt', 'class_iou.npy')) 
        class_threshold = np.load(args.labeled_id_path.replace('labeled.txt', 'class_threshold.npy')) 

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    if cfg['iou_samp']:
        if args.labeled_id_path.split('/')[-3] == 'pascal':
            if args.labeled_id_path.split('/')[-2] =='1_2':
                nsample_l = 10582-5291
            elif args.labeled_id_path.split('/')[-2] =='1_4':
                nsample_l = 10582-2646
            elif args.labeled_id_path.split('/')[-2] =='1_8':
                nsample_l = 10582-1323
            elif args.labeled_id_path.split('/')[-2] =='1_16':
                nsample_l = 10582-662
            else:
                nsample_l = 10582-int(args.labeled_id_path.split('/')[-2])
                
        elif args.labeled_id_path.split('/')[-3] == 'cityscapes':
            if args.labeled_id_path.split('/')[-2] =='1_2':
                nsample_l = 2976-1488
            elif args.labeled_id_path.split('/')[-2] =='1_4':
                nsample_l = 2976-744
            elif args.labeled_id_path.split('/')[-2] =='1_8':
                nsample_l = 2976-372
            elif args.labeled_id_path.split('/')[-2] =='1_16':
                nsample_l = 2976-186
                
        trainset_l = SemiDataset_OSamp(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['crop_size'], args.labeled_id_path, nsample=nsample_l, class_iou= class_iou)
        trainset_u = SemiDataset_OSamp(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['crop_size'], args.unlabeled_id_path, nsample=len(trainset_l.ids))
        
    else:
        trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
        trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
        
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    
    if rank == 0:
        logger.info('trainset_l : {}'.format(len(trainset_l)))
        logger.info('trainset_u : {}'.format(len(trainset_u)))
        logger.info('valset : {}'.format(len(valset)))
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.8f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()


            if cfg['conf_thre']: 
                with torch.no_grad():
                    model.eval()

                    pred_u_w_mix = model(img_u_w_mix).detach()
                    conf_u_w_mix_soft = pred_u_w_mix.softmax(dim=1)
                    conf_u_w_mix_thresh = threshold_pseudo(conf_u_w_mix_soft, class_threshold).cuda()
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                    img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

                pred_u_w = pred_u_w.detach()

                conf_u_w_soft = pred_u_w.softmax(dim=1)
                conf_u_w_thresh = threshold_pseudo(conf_u_w_soft, class_threshold).cuda()

                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)


                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, conf_u_w_cutmixed1_thresh = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), conf_u_w_thresh.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, conf_u_w_cutmixed2_thresh = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), conf_u_w_thresh.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1_thresh[cutmix_box1 == 1] = conf_u_w_mix_thresh[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2_thresh[cutmix_box2 == 1] = conf_u_w_mix_thresh[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

                loss_x = criterion_l(pred_x, mask_x)

                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1_thresh & (ignore_mask_cutmixed1 != 255))
                loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2_thresh & (ignore_mask_cutmixed2 != 255))
                loss_u_s2 = torch.sum(loss_u_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * (conf_u_w_thresh & (ignore_mask != 255))
                loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

                
            else:  ## baseline
                with torch.no_grad():
                    model.eval()
                    pred_u_w_mix = model(img_u_w_mix).detach()
                    conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                    img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)
                
                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

                loss_x = criterion_l(pred_x, mask_x)

                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                loss_u_s2 = torch.sum(loss_u_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
                loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

                
                
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += (loss_u_s1.item() + loss_u_s2.item()) / 2.0
            total_loss_w_fp += loss_u_w_fp.item()
            if cfg['conf_thre']: 
                total_mask_ratio += ((conf_u_w_thresh) & (ignore_mask != 255)).sum().item() / \
                                    (ignore_mask != 255).sum().item()
            else:
                total_mask_ratio += ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                                    (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_x / (i+1), total_loss_s / (i+1),
                    total_loss_w_fp / (i+1), total_mask_ratio / (i+1)))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for i_num,cl_name in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{} {}] IoU: {}'.format(i_num, class_name[i_num], cl_name))
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()

