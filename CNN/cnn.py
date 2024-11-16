import torch
from torch import nn
from einops.layers.torch import Rearrange
import math

class CNN_CLAS(nn.Module):
    def __init__(self, *,
                 img_size = (23, 23),
                 channels = 3, 
                 num_classes = 10, 
                 depth = 3, 
                 num_kernels = 3, 
                 conv_kernel_size = (3, 3), 
                 pool_kernel_size = (2, 2), 
                 padding = (1, 1), 
                 conv_stride = 1, 
                 pool_stride = 2):
        
        super().__init__()

        # Convolution Layers 
        self.conv_layers = nn.ModuleList()
        for _ in range(depth):
            kernel = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=conv_kernel_size, stride=conv_stride, padding=padding)
            conv_layer = nn.ModuleList([kernel for _ in range(num_kernels)])
            self.conv_layers.append(conv_layer)
            channels = num_kernels
        
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Feed Forward Layer
        features = channels * math.floor(img_size[0] / (pool_kernel_size[0] ** depth)) * math.floor(img_size[1] / (pool_kernel_size[1] ** depth))
        ff_layer = nn.Linear(in_features=features, out_features=num_classes)
        self.ff = nn.Sequential(Rearrange("b c h w -> b (c h w)"), ff_layer)

    def forward(self, x):
        # X (b c h w)
        for conv_layer in self.conv_layers:
            kernels = [kernel(x) for kernel in conv_layer]
            x = torch.cat(kernels, dim=1)
            x = self.relu(x)
            x = self.max_pool(x)
        
        out = self.ff(x)
        return out