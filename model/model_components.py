import torch
import torch.nn as nn
import torch.nn.functional as F


#### ===== basic Unet ===== ####
class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, ks=3, stride=1, padding=1, gpus=False, dropout_val=0.001):
        super(UnetConv, self).__init__()
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
        x = self.conv(inputs)
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


#### ===== Context Unet ===== #####
class SimpleUnetConv(nn.Module):
    def __init__(self, in_size, out_size, ks=3, stride=2, padding=1, gpus=False, dropout_val=0.001):
        super(SimpleUnetConv, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True))
        if gpus:
            self.conv.cuda()

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class SimpleUnetUp(nn.Module):
    def __init__(self, in_size, out_size, gpus=False):
        super(SimpleUnetUp, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1),
        )
        if gpus:
            self.up.cuda()

    def forward(self, feature):
        return self.up(feature)


class ContextModule(nn.Module):
    def __init__(self, in_size, out_size, gpus=False):
        super(ContextModule, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size, out_size, 3, padding=1)
        )
        if gpus:
            self.conv.cuda()

    def forward(self, x):
        return self.conv(x)


class Localization(nn.Module):
    def __init__(self, in_size, out_size, dropout_val=0, gpus = False):
        super(Localization, self).__init__()
        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, kernel_size=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))
        if gpus:
            self.conv.cuda()

    def forward(self, x):
        return self.conv(x)

class SegmentationLayer(nn.Module):
    def __init__(self, x_size, y_size, z_zize, gpus=False):
        super(SegmentationLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=x_size, out_channels=x_size, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=y_size, out_channels=y_size, kernel_size=1)
        self.up1 = SimpleUnetUp(x_size, y_size)
        self.up2 = SimpleUnetUp(y_size, z_zize)
        if gpus:
            self.conv1.cuda()
            self.conv2.cuda()
            self.up1.cuda()
            self.up2.cuda()

    def forward(self, x, y, z):
        con = self.conv1(x)
        up1 = self.up1(con)
        comb = up1 + self.conv2(y)
        return self.up2(comb) + z




#### ==== Spatial Transformer kinda doesn't do anything ==== ####
class SPTnet(nn.Module):
    def __init__(self, in_size, out_size, val, stride=1, ks=3, dropout_val=0, gpus=False):
        super(SPTnet, self).__init__()

        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))

        # spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_size, 32, kernel_size=7),  # 256*256*3
            nn.MaxPool2d(2, stride=2),  # 250*250*8
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),  # 125*125*8
            nn.MaxPool2d(2, stride=2),  # 121*121*16
            nn.ReLU(True)  # 60*60*16
        )
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(val ** 2, 16),
            nn.Linear(16, 3 * 2)
        )
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def rev_stn(self, x, theta: torch.tensor):
        empty_tensor = torch.empty(theta.size()).cuda()
        for val, batch in enumerate(theta):
            new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]]).cuda()), 0)
            theta = new_theta.inverse().cuda()
            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x, theta = self.stn(x)
        x = self.conv(x)
        x = self.rev_stn(x, theta)
        return x
