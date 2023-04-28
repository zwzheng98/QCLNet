r""" Implementation of center-pivot 4D convolution """

import torch
import torch.nn as nn


class SepConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(SepConv4d, self).__init__()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        # self.conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), stride=(1, 1),
        #                        bias=bias, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])
        self.gn2 = nn.GroupNorm(4, out_channels)
        self.gn1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        # if self.stride[2:][-1] > 1:
        #     out1 = self.prune(x)
        # else:
        #     out1 = x
        bsz, inch, ha, wa, hb, wb = x.size()
        out = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out = self.conv2(out)
        outch, o_hb, o_wb = out.size(-3), out.size(-2), out.size(-1)
        out = out.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        out = self.gn2(out)
        out = self.relu(out)

        out = out.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, outch, ha, wa)
        out = self.conv1(out)
        outch, o_ha, o_wa = out.size(-3), out.size(-2), out.size(-1)
        out = out.view(bsz, o_hb, o_wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        y = out
        return y
