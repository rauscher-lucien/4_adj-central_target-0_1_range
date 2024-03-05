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
    '''Pooling => (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)
     

class CenterBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(CenterBlock, self).__init__()
		self.convblock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
			ConvBlock(in_ch, out_ch),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		)

	def forward(self, x):
		return self.convblock(x)


class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			ConvBlock(in_ch, out_ch),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		)

	def forward(self, x):
		return self.convblock(x)


class OutConvBlock(nn.Module):
    """
    This block applies two convolutional layers followed by batch normalization and ReLU activation for the first two,
    and ends with a tanh activation function after the final convolution.
    """
    def __init__(self, in_ch, out_ch):
        super(OutConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            # nn.Tanh()  # Use Tanh activation function
        )

    def forward(self, x):
        return self.convblock(x)


class NewUNet(nn.Module):
    def __init__(self):
        super(NewUNet, self).__init__()

        self.base = 64

        # Define the network layers using the base size
        self.inc = ConvBlock(in_ch=4, out_ch=self.base)
        self.downc0 = DownBlock(in_ch=self.base, out_ch=2*self.base)
        self.downc1 = DownBlock(in_ch=2*self.base, out_ch=4*self.base)
        self.downc2 = DownBlock(in_ch=4*self.base, out_ch=8*self.base)
        self.centerc = CenterBlock(in_ch=8*self.base, out_ch=8*self.base)
        self.upc3 = UpBlock(in_ch=16*self.base, out_ch=4*self.base)
        self.upc2 = UpBlock(in_ch=8*self.base, out_ch=2*self.base)
        self.upc1 = UpBlock(in_ch=4*self.base, out_ch=self.base)
        self.outc = OutConvBlock(in_ch=2*self.base, out_ch=1)

        # Initialize weights
        self.apply(self.weight_init)

    def forward(self, x):
        # Forward pass through the UNet model
        tensor_to_image(x, 'image.png')
        x0 = self.inc(x)
        tensor_to_image(x0, 'image0.png')
        x1 = self.downc0(x0)
        tensor_to_image(x1, 'image1.png')
        x2 = self.downc1(x1)
        tensor_to_image(x2, 'image2.png')
        x3 = self.downc2(x2)
        tensor_to_image(x3, 'image3.png')
        x4 = self.centerc(x3)
        tensor_to_image(x4, 'image4.png')
        x3 = self.upc3(torch.cat([x3, x4], dim=1))
        tensor_to_image(x3, 'image5.png')
        x2 = self.upc2(torch.cat([x2, x3], dim=1))
        tensor_to_image(x2, 'image6.png')
        x1 = self.upc1(torch.cat([x1, x2], dim=1))
        tensor_to_image(x1, 'image7.png')
        x = self.outc(torch.cat([x0, x1], dim=1))
        tensor_to_image(x, 'image8.png')
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize with Kaiming initialization
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)