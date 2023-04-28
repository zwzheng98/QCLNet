
import torch.nn as nn
import torch.nn.functional as F
import torch
from .base.quaternion_layers import QuaternionConv as Qconv
from .base.quaternion_layers import QuaternionTransposeConv as QTconv
from .base.sepconv4d import SepConv4d as Conv4d
from .base.Q2RM import Q2RM

from .base.QN import QuaternionNorm2d as QN

class QCLLearner(nn.Module):
    def __init__(self, inch):
        super(QCLLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            # assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        def make_building_qconv(channels):

            building_block_layers = []
            for i in range(len(channels) - 1):
                building_block_layers.append(Qconv(channels[i] * 4, channels[i + 1] * 4, 3, 1, padding=1))
                building_block_layers.append(QN(channels[i + 1] * 4))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4, outch5 = 16, 32, 64, 64, 64

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_qconv([outch5, outch5, outch5, outch5])
        self.encoder_layer3to2 = make_building_qconv([outch5, outch5, outch5, outch5])

        # quaternion layers
        self.quaternion_encoder_layer4 = make_building_qconv([outch3, outch4, outch4, outch5])
        self.quaternion_encoder_layer3 = make_building_qconv([outch3, outch4, outch4, outch5])
        self.quaternion_encoder_layer2 = make_building_qconv([outch3, outch4, outch4, outch5])

        self.decoder1 = nn.Sequential(nn.Conv2d(64+32, 48, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv2d(48+16, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())
        self.cls = nn.Sequential(nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

        self.Q2RM = Q2RM(128, M=4)
        self.dropout2d = nn.Dropout2d(p=0.5)

        self.proj_query_feat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 32, 1),
                nn.ReLU(),
                nn.Dropout2d(p=0.5)
            ),
            nn.Sequential(
                nn.Conv2d(256, 16, 1),
                nn.ReLU(),
                nn.Dropout2d(p=0.5)
            )
        ])

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode='nearest') for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [F.interpolate(x, size=size, mode='nearest') for x, size in zip(feats, sizes)]
        return recoverd_feats

    def forward(self, hypercorr_pyramid, query_feats):

        query_feat5, query_feat4, query_feat3, query_feat2 = self.extract_last(query_feats)
        query_feat3 = F.interpolate(query_feat3, (60, 60), mode='bilinear', align_corners=True)
        query_feat2 = F.interpolate(query_feat2, (120, 120), mode='bilinear', align_corners=True)

        query_feat3, query_feat2 = [
            self.proj_query_feat[i](x) for i, x in enumerate((query_feat3, query_feat2))
        ]

        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        bsz, ch, ha, wa, hb, wb = hypercorr_sqz4.size()
        hypercorr_sqz4 = hypercorr_sqz4.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz, -1, ha, wa)

        bsz, ch, ha, wa, hb, wb = hypercorr_sqz3.size()
        hypercorr_sqz3 = hypercorr_sqz3.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz, -1, ha, wa)

        bsz, ch, ha, wa, hb, wb = hypercorr_sqz2.size()
        hypercorr_sqz2 = hypercorr_sqz2.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz, -1, ha, wa)

        hypercorr_sqz4 = self.quaternion_encoder_layer4(hypercorr_sqz4)
        hypercorr_sqz3 = self.quaternion_encoder_layer3(hypercorr_sqz3)
        hypercorr_sqz2 = self.quaternion_encoder_layer2(hypercorr_sqz2)

        hypercorr_sqz4 = F.interpolate(hypercorr_sqz4, hypercorr_sqz3.size()[-2:], mode='bilinear', align_corners=True)
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = F.interpolate(hypercorr_mix43, hypercorr_sqz2.size()[-2:], mode='bilinear',
                                       align_corners=True)
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        hypercorr_encoded = self.Q2RM(hypercorr_mix432)

        hypercorr_decoded = self.decoder1(torch.cat((hypercorr_encoded, query_feat3), dim=1))
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        hypercorr_decoded = self.decoder2(torch.cat((hypercorr_decoded, query_feat2), dim=1))
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)

        logit_mask = self.cls(hypercorr_decoded)

        return logit_mask
