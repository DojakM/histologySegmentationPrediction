from typing import Any

import torch
import torch.nn as nn

from model.model_components import UnetConv, UnetUp
from model.unet_super import UnetSuper
from utils import weights_init


class Unet(UnetSuper):
    def __init__(self, len_test_set, hparams, input_channels, is_deconv=True,
                 is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__(len_test_set=len_test_set, hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [32, 64, 128, 256]
        self.conv1 = UnetConv(self.in_channels, filters[0], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.center = UnetConv(filters[2], filters[3], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv1(inputs)  # 16*256*256
        maxpool1 = maxpool(conv1)  # 16*128*128

        conv2 = self.conv2(maxpool1)  # 32*128*128
        maxpool2 = maxpool(conv2)  # 32*64*64

        conv3 = self.conv3(maxpool2)  # 64*64*64
        maxpool3 = maxpool(conv3)  # 64*32*32

        center = self.center(maxpool3)

        up3 = self.up_concat3(center, conv3)  # 64*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)
        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)
