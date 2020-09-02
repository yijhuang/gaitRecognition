import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock3d(nn.Module):
    def __init__(self, forward_block, pooling=False, **kwargs):
        super(SetBlock3d, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool3d = nn.MaxPool3d(**kwargs)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        #x_hw = self.forward_block(x)
        #x_th = self.forward_block(x.permute(0,1,4,2,3).contiguous()).permute(0,1,3,4,2).contiguous()
        #x_tw = self.forward_block(x.permute(0,1,3,2,4).contiguous()).permute(0,1,3,2,4).contiguous()
        #print(x_hw.shape,x_th.shape,x_tw.shape)
        #input()
        #x = torch.cat([x_hw,x_tw,x_th])
        x = self.forward_block(x)
        #print(x.shape) 
        #input()
        if self.pooling: 
            x = self.pool3d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)
        return x
