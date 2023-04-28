r""" QCLNet """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import QCLLearner


class QCLNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(QCLNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(3, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.QCL_learner = QCLLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                               self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats[-self.stack_ids[-1]:],
                                                      support_feats[-self.stack_ids[-1]:], self.stack_ids)
        index = 0
        b_s, ch, ha, wa, hb, wb = corr[index].size()
        corr_matrix = corr[index].view(b_s, ch, ha, wa, hb*wb).max(4)[0]

        corr_matrix = corr_matrix[:, -1, :, :].unsqueeze(1)
        corr_matrix = F.sigmoid(corr_matrix)

        corr_matrix = F.interpolate(corr_matrix, support_img.size()[2:], mode='bilinear', align_corners=True)

        logit_mask = self.QCL_learner(corr, self.stack_feats(query_feats))
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)
            # corr_matrix = F.interpolate(corr_matrix, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit_mask, corr_matrix.squeeze(1)

    def stack_feats(self, feats):

        feats_l4 = torch.stack(feats[-self.stack_ids[0]:]).transpose(0, 1)
        feats_l3 = torch.stack(feats[-self.stack_ids[1]:-self.stack_ids[0]]).transpose(0, 1)
        feats_l2 = torch.stack(feats[-self.stack_ids[2]:-self.stack_ids[1]]).transpose(0, 1)
        feats_l1 = torch.stack(feats[:-self.stack_ids[2]]).transpose(0, 1)

        return [feats_l4, feats_l3, feats_l2, feats_l1]

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear',
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):
        pred_mask_list = []
        corr_matrix_list = []
        for s_idx in range(nshot):
            logit_mask, corr_matrix = self(batch['query_img'], batch['support_imgs'][:, s_idx],
                                           batch['support_masks'][:, s_idx])
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
            pred_mask_list.append(logit_mask.argmax(dim=1))
            if s_idx == 0:
                corr_matrixs = corr_matrix.unsqueeze(0)
                logit_masks = logit_mask.unsqueeze(0)
            else:
                corr_matrixs = torch.cat([corr_matrixs, corr_matrix.unsqueeze(0)], dim=0)
                logit_masks = torch.cat([logit_masks, logit_mask.unsqueeze(0)], dim=0)

        corr_matrixs = F.softmax(corr_matrixs, dim=0)

        for i in range(5):
            corr_matrix_list.append(corr_matrixs[i, :, :, :])
        logit_mask_agg = (logit_masks.argmax(dim=2).clone()) * corr_matrixs
        logit_mask_agg = logit_mask_agg.sum(dim=0)

        pred_mask = logit_mask_agg.clone()
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()
