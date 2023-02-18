import torch
import torch.nn as nn


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1,gpus=False,dropout_val=0):
        super(UnetConv, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)))
        else:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True)))
        if gpus:
            self.conv.cuda()
    def forward(self, inputs):
        conv = self.conv
        x = conv(inputs)
        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, gpus=False, dropout_val=0):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_size, out_size, False)
        self.up = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1)
        )
        if gpus:
            self.conv.cuda()
            self.up.cuda()

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        return self.conv(outputs0)

    def print(self, *kwargs):
        print(kwargs)

