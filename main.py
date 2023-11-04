import torch.nn as nn
import torch
from torchinfo import summary
import torchvision.transforms as transforms

from src.u_net import UNet

model = UNet()
print(summary(model, (1, 1, 572, 572)))