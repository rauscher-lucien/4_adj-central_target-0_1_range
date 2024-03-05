import torch
import torch.nn as nn

from utils import *


class ConvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
    

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a =  self.convblock(x)
        x_b = self.pool(x_a)
        return x_a, x_b


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock(2*in_ch, out_ch)

    def forward(self, x, x_new):
        x = self.upsample(x)
        x = torch.cat([x, x_new], dim=1)
        x = self.convblock(x)
        return x



class NewUNet(nn.Module):
    def __init__(self):
        super(NewUNet, self).__init__()

        self.base = 32

        # Define the network layers using the base size
        self.down0 = DownBlock(in_ch=4, out_ch=self.base)
        self.down1 = DownBlock(in_ch=self.base, out_ch=2*self.base)
        self.down2 = DownBlock(in_ch=2*self.base, out_ch=4*self.base)
        self.down3 = DownBlock(in_ch=4*self.base, out_ch=8*self.base)
        self.down4 = DownBlock(in_ch=8*self.base, out_ch=16*self.base)
        self.conv = ConvBlock(in_ch=16*self.base, out_ch=16*self.base)
        self.up5 = UpBlock(in_ch=16*self.base, out_ch=8*self.base)
        self.up4 = UpBlock(in_ch=8*self.base, out_ch=4*self.base)
        self.up3 = UpBlock(in_ch=4*self.base, out_ch=2*self.base)
        self.up2 = UpBlock(in_ch=2*self.base, out_ch=self.base)
        self.up1 = UpBlock(in_ch=self.base, out_ch=self.base)
        self.outconv = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, bias=False)

        # Initialize weights
        self.apply(self.weight_init)

    def forward(self, x):
        # Forward pass through the UNet model
        #tensor_to_image(x, 'image0.png')
        x0_a, x0_b = self.down0(x)
        #tensor_to_image(x0_b, 'image1.png')
        x1_a, x1_b = self.down1(x0_b)
        #tensor_to_image(x1_b, 'image2.png')
        x2_a, x2_b = self.down2(x1_b)
        #tensor_to_image(x2_b, 'image3.png')
        x3_a, x3_b = self.down3(x2_b)
        #tensor_to_image(x3_b, 'image4.png')
        x4_a, x4_b = self.down4(x3_b)
        #tensor_to_image(x4_b, 'image5.png')
        x5 = self.conv(x4_b)
        #tensor_to_image(x5, 'image6.png')
        x4 = self.up5(x5, x4_a)
        #tensor_to_image(x4, 'image7.png')
        x3 = self.up4(x4, x3_a)
        #tensor_to_image(x3, 'image8.png')
        x2 = self.up3(x3, x2_a)
        #tensor_to_image(x2, 'image9.png')
        x1 = self.up2(x2, x1_a)
        #tensor_to_image(x1, 'image10.png')
        x0 = self.up1(x1, x0_a)
        #tensor_to_image(x0, 'image11.png')
        x = self.outconv(x0)
        #tensor_to_image(x, 'image12.png')
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize with Kaiming initialization
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)