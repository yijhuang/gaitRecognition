import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)
        #return x

class RAL(nn.Module):
    def __init__(self, in_channels, frame_num, **kwargs):
        super(RAL, self).__init__()
        self.conv1_1 = BasicConv2d(in_channels, 1, [3,3], padding=1)
        #self.conv1_1 = nn.Conv2d(in_channels, 1, [3,3], bias=False, **kwargs)
        self.conv1_2 = BasicConv2d(1, 1, [1,1])
        #self.conv1_2 = nn.Conv2d(1, 1, [1,1], bias=False, **kwargs)
        self.conv2_1 = BasicConv2d(in_channels, int(in_channels/16), [1,1])                                 
        #self.conv2_1 = nn.Conv2d(in_channels, int(in_channels/16), [1,1], bias=False, **kwargs)
        self.conv2_2 = BasicConv2d(int(in_channels/16), in_channels, [1,1])
        #self.conv2_2 = nn.Conv2d(int(in_channels/16), in_channels, [1,1], bias=False, **kwargs)
        self.conv3_1 = BasicConv2d(frame_num, frame_num, [1,1])
        #self.conv3_1 = nn.Conv2d(in_channels, frame_num, [1,1], bias=False, **kwargs)
        self.conv3_2 = BasicConv2d(frame_num, frame_num, [1,1])
        #self.conv3_2 = nn.Conv2d(frame_num, frame_num, [1,1], bias=False, **kwargs)
    def forward(self,x):

        #_, c, n, h, w = x.size()
        _, n, c, h, w = x.size()

        x_SAL = self.conv1_2(self.conv1_1(torch.mean(x,1)))
        x_SAL = x_SAL.unsqueeze(1)

        x_CAL = self.conv2_2(self.conv2_1(torch.mean(torch.mean(torch.mean(x,1),2,keepdim=True),3,keepdim=True)))
        x_CAL = x_CAL.unsqueeze(1)

        x_TAL = self.conv3_2(self.conv3_1(torch.mean(torch.mean(torch.mean(x,2),2,keepdim=True),3,keepdim=True)))
        x_TAL = x_TAL.unsqueeze(2)

        x_RAL = torch.sigmoid(torch.mul(torch.mul(x_SAL,x_TAL),x_CAL)).mul(x)
  
        return 0.5*x + x_RAL   