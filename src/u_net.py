import torch.nn as nn
import torch
from torchinfo import summary
import torchvision.transforms as transforms


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.norm(self.conv(x)))
        return x
    

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.pool(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.batch_norm(self.upconv(x))
    
class CopyCrop(nn.Module):
    def __init__(self, h, w):
        super(CopyCrop, self).__init__()

        self.crop = transforms.CenterCrop((h, w))
        
    def forward(self, x):
        return self.crop(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # downsampling
        self.down = DownSample()
        self.conv1 = nn.Sequential(Conv3x3(1, 64), Conv3x3(64, 64))
        self.conv2 = nn.Sequential(Conv3x3(64, 128), Conv3x3(128, 128))
        self.conv3 = nn.Sequential(Conv3x3(128, 256), Conv3x3(256, 256))
        self.conv4 = nn.Sequential(Conv3x3(256, 512), Conv3x3(512, 512))

        # bottom 
        self.bottom = nn.Sequential(Conv3x3(512, 1024), Conv3x3(1024, 1024))

        # upsampling 
        self.upconv4 = nn.Sequential(UpSample(1024, 512))
        self.upconv3 = nn.Sequential(UpSample(512, 256))
        self.upconv2 = nn.Sequential(UpSample(256, 128))
        self.upconv1 = nn.Sequential(UpSample(128, 64))
        self.conv4_ = nn.Sequential(Conv3x3(1024, 512), Conv3x3(512, 512))
        self.conv3_ = nn.Sequential(Conv3x3(512, 256), Conv3x3(256, 256))
        self.conv2_ = nn.Sequential(Conv3x3(256, 128), Conv3x3(128, 128))
        self.conv1_ = nn.Sequential(Conv3x3(128, 64), Conv3x3(64, 64))

        self.copy_crop4 = CopyCrop(56, 56)
        self.copy_crop3 = CopyCrop(104, 104)
        self.copy_crop2 = CopyCrop(200, 200)
        self.copy_crop1 = CopyCrop(392, 392)

        self.conv1x1  = nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=True)
        self.norm = nn.BatchNorm2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down(x1))
        x3 = self.conv3(self.down(x2))
        x4 = self.conv4(self.down(x3))

        bottom = self.bottom(self.down(x4))

        in4 = torch.concat((self.copy_crop4(x4), self.upconv4(bottom)), dim=1)
        xx4 = self.conv4_(in4)
        in3 = torch.concat((self.copy_crop3(x3), self.upconv3(xx4)), dim=1)
        xx3 = self.conv3_(in3)
        in2 = torch.concat((self.copy_crop2(x2), self.upconv2(xx3)), dim=1)
        xx2 = self.conv2_(in2)
        in1 = torch.concat((self.copy_crop1(x1), self.upconv1(xx2)), dim=1)
        xx1 = self.conv1_(in1)

        out = self.norm(self.conv1x1(xx1))

        return out




