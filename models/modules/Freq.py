import torch
import torch.nn as nn
import numpy as np

class BlockAttention(nn.Module):
    def __init__(self, in_channels, block_scale, reduction_ratio=32):
        super(BlockAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.block_scale = block_scale
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.block_scale, self.block_scale, W // self.block_scale, self.block_scale)
        x = x.transpose(3, 4)
        x = x.reshape(B, -1, self.block_scale, self.block_scale)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = x * self.sigmoid(out)
        x = x.reshape(B, C, H // self.block_scale, W // self.block_scale,  self.block_scale, self.block_scale)
        x = x.transpose(3, 4)
        x = x.reshape(B, C, H, W)
        return x

class FreqAttention (nn.Module):
    def __init__(self, in_channel, in_size ,scale=8 ):
        super(FreqAttention, self).__init__()
        self.in_channel =in_channel
        self.in_size = in_size
        self.scale = scale
        self.DCT_mat = np.zeros(shape=(scale, scale)).astype(np.float32)
        for i in range(scale):
            c = 0
            if i == 0:
                c = np.sqrt(1 / scale)
            else:
                c = np.sqrt(2 / scale)
            for j in range(scale):
                self.DCT_mat[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * scale))
        self.DCT_mat = torch.from_numpy(self.DCT_mat)
        self.params = nn.Parameter(torch.rand(self.scale, self.scale))
        self.block_att = BlockAttention(self.in_channel* in_size//self.scale * in_size//self.scale, self.scale)

    def DCT(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
        x = x.transpose(3, 4)
        x = torch.matmul(self.DCT_mat.to(x.device), x)
        x = torch.matmul(x, self.DCT_mat.t().to(x.device))
        x = x.transpose(3, 4)
        x = x.reshape(B, C, H, W)
        return x

    def IDCT(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
        x = x.transpose(3, 4)
        x = torch.matmul(self.DCT_mat.t().to(x.device), x)
        x = torch.matmul(x, self.DCT_mat.to(x.device))
        x = x.transpose(3, 4)
        x = x.reshape(B, C, H, W)
        return x

    def freq_att(self, x):
        x = self.block_att(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
        x = x.transpose(3, 4)
        x = x * self.params
        x = x.transpose(3, 4)
        x = x.reshape(B, C, H, W)
        return x

    def forward(self, x):
        x = self.DCT(x)
        x = self.freq_att(x)
        x = self.IDCT(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    x = torch.rand(2, 4, 80, 80)
    freq = FreqAttention()
    y = freq(x)
    print(y.shape)
