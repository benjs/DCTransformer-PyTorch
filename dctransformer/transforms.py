import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter

class RGBToYCbCr(nn.Module):
    """Converts a tensor from RGB to YCbCr color space.
    Using transform from https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/jccolor.c
    """
    def __init__(self):
        super().__init__()
        
        transform = torch.tensor([
            [ 0.299000,  0.587000,  0.114000],
            [-0.168736, -0.331264,  0.500000],
            [ 0.500000, -0.418688, -0.081312]
        ])

        transform_bias = torch.tensor([0, 0.5, 0.5])

        self.register_buffer('transform', transform[:, :, None, None], persistent=False)
        self.register_buffer('transform_bias', transform_bias, persistent=False)

    def forward(self, x:torch.Tensor):
        return F.conv2d(x, self.transform, self.bias)
