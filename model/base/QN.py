# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 00:39:36 2020

@author: Edoardo
"""

import torch
from torch.nn import Module, Parameter


# import math

def moving_average_update(statistic, curr_value, momentum):
    new_value = (1 - momentum) * statistic + momentum * curr_value

    return new_value.data


class QuaternionNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, num_groups=4, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)
        self.num_groups = num_groups

        # self.register_buffer('moving_var', torch.ones(8, self.num_groups, 1))
        # self.register_buffer('moving_mean', torch.zeros(4, 8, self.num_groups, 1))
        self.momentum = momentum

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        G = self.num_groups

        b_s, ch, ha, wa = input.size()
        input = input.view(b_s, 4, int(ch / 4), ha, wa)

        x = input.view(b_s, G, -1)
        mean = x.mean(-1, keepdim=True)

        delta = x - mean  # b 4
        quat_variance = (delta ** 2).mean(dim=2).mean(dim=1)
        quat_variance = quat_variance.unsqueeze(1).unsqueeze(1)
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        normalized = delta / denominator
        normalized = normalized.view(b_s, ch, ha, wa)

        gamma = self.gamma.repeat(1, 4, 1, 1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        output = (gamma * normalized) + self.beta

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
