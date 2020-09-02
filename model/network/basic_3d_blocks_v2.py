#CoST new 3D convolution Construction
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv3d, self).__init__()
        #self.conv_weight = nn.ParameterList([
        #    nn.Parameter(nn.init.xavier_uniform_(torch.empty(kernel_size)))])
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x, y, softmax):
        #stride_t,stride_h,stride_w = self.stride
        x_hw = self.conv(x)
        x_tw = self.conv(x.transpose(2,3).contiguous()).transpose(2,3).contiguous()
        x_th = self.conv(x.transpose(2,4).contiguous()).transpose(2,4).contiguous()

        x_hw = x_hw.unsqueeze(2)
        x_tw = x_tw.unsqueeze(2)
        x_th = x_th.unsqueeze(2)
        x = torch.cat([x_hw,x_th,x_tw], 2)

        b,c,_,n,h,w = x.size()
        x = x.view(b,c,_,-1)
        x = softmax(y[0]).matmul(x)
        x = x.view(b,c,n,h,w) 

        return x


class SetBlock3d(nn.Module):
    def __init__(self, forward_block, pooling=False, **kwargs):
        super(SetBlock3d, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool3d = nn.MaxPool3d(**kwargs)
    def forward(self, x, y, softmax):
        #n, c, t, h, w = x.size()
        #x = self.forward_block(x.view(-1,c,h,w))
        x = self.forward_block(x,y,softmax)
        #print(x_hw.shape,x_th.shape,x_tw.shape)
        #input()
        #x = torch.cat([x_hw,x_tw,x_th])
        if self.pooling: 
            x = self.pool3d(x)
        #_, c, h, w = x.size()
        #return x.view(n, s, c, h ,w)
        return x
