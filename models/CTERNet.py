from __future__ import print_function
from __future__ import division

import torch
import math
import torch.nn as nn
from torch import einsum
from math import log
import torch.optim as optim
from torch.optim import lr_scheduler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from lib.CauLib.CAM import CAM_Module
from lib.Res2Net import res2net101_v1b_26w_4s
from torchvision import models

import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy
import cv2

from torch.autograd import Function, Variable

from .SE_weight_module import SEWeightModule

EPSILON = 1e-6


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.CAM = nn.Sequential(CAM_Module(32, 14))

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.CAM(feats)

        return out


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    # @staticmethod
    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print(
                'Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    # @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    # @staticmethod
    def forward(self, input):
        CWPF = ClassWisePoolFunction(self.num_maps)
        return CWPF(input)


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


def deconv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])

        self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        self.num_ftrs = model.fc.in_features
        self.downconv = nn.Sequential(
            nn.Conv2d(32, num_classes * num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1))
        self.conv6 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 32, kernel_size=1)
        )

        self.GAP = nn.AvgPool2d(14)
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(33, num_classes)
        )

        self.CAM2 = nn.Sequential(CAM_Module(32, 14))

        self.num_features = 2048
        self.M = 32
        self.bap = BAP(pool='GAP')

        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        self.fc = nn.Linear(self.M * self.num_features, num_classes, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cca = CrissCrossAttention(32)

        self.SituationSelection = PSAModule(2048, 64, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])

    # @staticmethod
    def forward(self, x):
        r1, r2, r3, r4 = self.resnet(x)
        r5 = self.conv6(r4)

        #Situation Slection
        attention_maps = self.SituationSelection(r4)
        attention_maps = self.conv5(attention_maps)
        attention_maps = self.CAM2(attention_maps)  # 32
        # print(attention_maps.size())

        #C^2RM Module
        attention_maps, counterfactual_map = self.bap(r5, attention_maps)

        # ```
        # Codes of class BAP
        #     will be released after acceptance
        # ```

        # detect branch
        x_ori = r5
        x1 = self.downconv(attention_maps)
        x2 = self.downconv(counterfactual_map)

        x_conv = x1
        x1 = self.GAP(x1)
        # print('after GAP x shape is ', x1.size())
        x1 = self.spatial_pooling(x1)
        # print('after GAP x shape is ', x1.size())
        x1 = x1.view(x1.size(0), -1)
        # print('after GAP x shape is ', x1.size())

        x2 = self.GAP(x2)  # x = self.GMP(x)
        # print('after GAP x shape is ', x.shape)
        x2 = self.spatial_pooling(x2)
        # print('after pooling x shape is ', x.shape)
        x2 = x2.view(x2.size(0), -1)
        # print(x.type())



        # cls branch
        x_conv = self.spatial_pooling(x_conv)

        x_conv = x_conv * x1.view(x1.size(0), x1.size(1), 1, 1)
        x_conv = self.spatial_pooling2(x_conv)
        # print(x_conv.size())

        x_conv_copy = x_conv.repeat(1, 32, 1, 1)
        # print(x_conv_copy.size()) #2047,14,14
        x_conv_copy = torch.mul(x_conv_copy, x_ori)

        x_conv_copy = torch.cat((x_conv, x_conv_copy), 1)
        x_conv_copy = self.GAP(x_conv_copy)
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0), -1)
        x_conv_copy = self.classifier(x_conv_copy)
        # print(x_conv_copy.type())

        x_conv2 = self.spatial_pooling(x2)

        x_conv2 = x_conv2 * x2.view(x2.size(0), x2.size(1), 1, 1)
        x_conv2 = self.spatial_pooling2(x_conv2)
        # print(x_conv.size())

        x_conv_copy2 = x_conv2.repeat(1, 32, 1, 1)
        # print(x_conv_copy.size()) #2047,14,14
        x_conv_copy2 = torch.mul(x_conv_copy2, x_ori)

        x_conv_copy2 = torch.cat((x_conv2, x_conv_copy), 1)
        x_conv_copy2 = self.GAP(x_conv_copy2)
        x_conv_copy2 = x_conv_copy2.view(x_conv_copy2.size(0), -1)
        x_conv_copy2 = self.classifier(x_conv_copy2)

        # The total effect
        YTE = x_conv_copy2 - x_conv_copy

        return x1, YTE
