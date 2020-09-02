import torch
import torch.nn as nn
import torch.nn.functional as F

class M3D_3(nn.Module):
    #def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(M3D_3, self).__init__()
        self.channels = int(in_channels/2)
        self.conv1 = nn.Conv3d(in_channels, self.channels, [1,1,1], bias=False, **kwargs)
        self.conv2 = nn.Conv3d(self.channels, self.channels, [1,3,3], padding=[0,1,1], bias=False, **kwargs)
        self.conv_tem1 = nn.Conv3d(self.channels, self.channels, [3,1,1], padding=[1,0,0], bias=False, **kwargs)
        self.conv_tem2 = nn.Conv3d(self.channels, self.channels, [5,1,1], padding=[2,0,0], bias=False, **kwargs)
        self.conv_tem3 = nn.Conv3d(self.channels, self.channels, [7,1,1], padding=[3,0,0], bias=False, **kwargs)
        self.conv3 = nn.Conv3d(self.channels, out_channels,[1,1,1], bias=False, **kwargs)
    def forward(self, x):
        x_tem = self.conv1(x)
        x_tem = self.conv2(x_tem)
        x_tem_1 = self.conv_tem1(x_tem)
        x_tem_2 = self.conv_tem2(x_tem)
        x_tem_3 = self.conv_tem3(x_tem)
        x_tem = self.conv3(x_tem + x_tem_1 + x_tem_2 + x_tem_3)
        x = x + x_tem
        return F.leaky_relu(x, inplace=True)

class M3D_2(nn.Module):
    #def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(M3D_2, self).__init__()
        self.channels = int(in_channels/2)
        self.conv1 = nn.Conv3d(in_channels, self.channels, [1,1,1], bias=False, **kwargs)
        self.conv2 = nn.Conv3d(self.channels, self.channels, [1,3,3], padding=[0,1,1], bias=False, **kwargs)
        self.conv_tem1 = nn.Conv3d(self.channels, self.channels, [3,1,1], padding=[1,0,0], bias=False, **kwargs)
        self.conv_tem2 = nn.Conv3d(self.channels, self.channels, [5,1,1], padding=[2,0,0],bias=False, **kwargs)
        self.conv3 = nn.Conv3d(self.channels, out_channels,[1,1,1], bias=False, **kwargs)
    def forward(self, x):
        x_tem = self.conv1(x)
        x_tem = self.conv2(x_tem)
        x_tem_1 = self.conv_tem1(x_tem)
        x_tem_2 = self.conv_tem2(x_tem)
        x_tem = self.conv3(x_tem + x_tem_1 + x_tem_2)
        x = x + x_tem
        return F.leaky_relu(x, inplace=True)
    
class M3D_1(nn.Module):
    #def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(M3D_1, self).__init__()
        self.channels = int(in_channels/2)
        self.conv1 = nn.Conv3d(in_channels, self.channels, [1,1,1], bias=False, **kwargs)
        self.conv2 = nn.Conv3d(self.channels, self.channels, [1,3,3], padding=[0,1,1], bias=False, **kwargs)
        self.conv_tem1 = nn.Conv3d(self.channels, self.channels, [3,1,1], padding=[1,0,0], bias=False, **kwargs)
        self.conv3 = nn.Conv3d(self.channels, out_channels,[1,1,1], bias=False, **kwargs)
    def forward(self, x):
        x_tem = self.conv1(x)
        x_tem = self.conv2(x_tem)
        x_tem_1 = self.conv_tem1(x_tem)
        x_tem = self.conv3(x_tem + x_tem_1)
        x = x + x_tem
        return F.leaky_relu(x, inplace=True)

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.BN = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.BN(F.relu(self.conv(x)))
        return x

# class RAL(nn.Module):
#     def __init__(self, in_channels, frame_num, **kwargs):
#         super(RAL, self).__init__()
#         self.conv1_1 = BasicConv2d(in_channels, 1, [3,3], padding=1)
#         #self.conv1_1 = nn.Conv2d(in_channels, 1, [3,3], bias=False, **kwargs)
#         self.conv1_2 = BasicConv2d(1, 1, [1,1])
#         #self.conv1_2 = nn.Conv2d(1, 1, [1,1], bias=False, **kwargs)
#         self.conv2_1 = BasicConv2d(in_channels, int(in_channels/16), [1,1])                                 
#         #self.conv2_1 = nn.Conv2d(in_channels, int(in_channels/16), [1,1], bias=False, **kwargs)
#         self.conv2_2 = BasicConv2d(int(in_channels/16), in_channels, [1,1])
#         #self.conv2_2 = nn.Conv2d(int(in_channels/16), in_channels, [1,1], bias=False, **kwargs)
#         self.conv3_1 = BasicConv2d(frame_num, frame_num, [1,1])
#         #self.conv3_1 = nn.Conv2d(in_channels, frame_num, [1,1], bias=False, **kwargs)
#         self.conv3_2 = BasicConv2d(frame_num, frame_num, [1,1])
#         #self.conv3_2 = nn.Conv2d(frame_num, frame_num, [1,1], bias=False, **kwargs)
#     def forward(self,x):
#         #_, c, n, h, w = x.size()
#         _, n, c, h, w = x.size()

#         x_SAL = self.conv1_2(self.conv1_1(torch.mean(x,1)))
#         x_SAL = x_SAL.unsqueeze(1)

#         x_CAL = self.conv2_2(self.conv2_1(torch.mean(torch.mean(torch.mean(x,1),2,keepdim=True),3,keepdim=True)))
#         x_CAL = x_CAL.unsqueeze(1)

#         x_TAL = self.conv3_2(self.conv3_1(torch.mean(torch.mean(torch.mean(x,2),2,keepdim=True),3,keepdim=True)))
#         x_TAL = x_TAL.unsqueeze(2)

#         x_RAL = torch.sigmoid(torch.mul(torch.mul(x_SAL,x_TAL),x_CAL)).mul(x)

#         return 0.5*x + x_RAL

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
        _, c, n, h, w = x.size()
        #_, n, c, h, w = x.size()

        x_SAL = self.conv1_2(self.conv1_1(torch.mean(x,2)))
        x_SAL = x_SAL.unsqueeze(2)

        x_CAL = self.conv2_2(self.conv2_1(torch.mean(torch.mean(torch.mean(x,2),2,keepdim=True),3,keepdim=True)))
        x_CAL = x_CAL.unsqueeze(2)

        x_TAL = self.conv3_2(self.conv3_1(torch.mean(torch.mean(torch.mean(x,1),2,keepdim=True),3,keepdim=True)))
        x_TAL = x_TAL.unsqueeze(1)

        x_RAL = torch.sigmoid(torch.mul(torch.mul(x_SAL,x_TAL),x_CAL)).mul(x)

        return 0.5*x + x_RAL   
    
class SetBlock3d(nn.Module):
    def __init__(self, forward_block, pooling=False, **kwargs):
        super(SetBlock3d, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool3d = nn.MaxPool3d(**kwargs)
    def forward(self, x):
        #n, c, t, h, w = x.size()
        #x = self.forward_block(x.view(-1,c,h,w))
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
        #_, c, h, w = x.size()
        #return x.view(n, s, c, h ,w)
        return x
