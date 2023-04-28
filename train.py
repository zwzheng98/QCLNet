r""" QCLNet training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.QCLNet import QCLNet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    # lr = base_lr
    # print(base_lr, lr)

    if curr_iter % 500 == 0:
        print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        # print(index)
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr
            # print("current lr is :", lr * scale_lr)

def train(epoch, model, dataloader, optimizer, training):
    r""" Train QCLNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    max_iter = args.niter * len(dataloader)
    for idx, batch in enumerate(dataloader):
        # current_iter = epoch * len(dataloader) + idx + 1
        # if args.lr > 1e-6 and training:
        #     poly_learning_rate(optimizer, args.lr, current_iter, max_iter)
        # 1. QCLNet forward pass
        batch = utils.to_cuda(batch)
        logit_mask, _ = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='QCLNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3) # 0.001
    parser.add_argument('--niter', type=int, default=500)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = QCLNet(args.backbone, False)
    Logger.log_params(model)

    # Device setup CUDA_VISIBLE_DEVICES=0 python train.py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    # optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}])
    # optimizer = optim.SGD([{"params": model.parameters(),
    #                         "lr": args.lr, "weight_decay:": args.weight_decay, "momentum": args.momentum}])
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(benchmark=args.benchmark, img_size=473, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train QCLNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
