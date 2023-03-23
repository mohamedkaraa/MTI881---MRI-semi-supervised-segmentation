from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from UNet_Base import initialize_weights


class conv_block(nn.Module):
    '''
    Convolution Block 
    '''
    
    def __init__(self, in_channels, out_channels,dropout=False):
        super(conv_block, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    '''
    Up Convolution Block
    '''
    
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, num_classes):
        super(AttU_Net, self).__init__()

        self.Conv1 = conv_block(1, 16)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = conv_block(16, 32)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = conv_block(32, 64)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = conv_block(64, 128)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv5 = conv_block(128, 256,True)

        self.Up5 = up_conv(256, 128)
        self.Att5 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv5 = conv_block(256, 128)

        self.Up4 = up_conv(128, 64)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv4 = conv_block(128, 64)

        self.Up3 = up_conv(64, 32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv3 = conv_block(64, 32)

        self.Up2 = up_conv(32, 16)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=32)
        self.Up_conv2 = conv_block(32, 16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        initialize_weights(self)


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.final(d2)

        return out