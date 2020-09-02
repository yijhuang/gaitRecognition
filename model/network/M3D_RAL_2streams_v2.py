import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from .M3D import SetBlock3d, BasicConv3d, M3D_3, M3D_2, M3D_1, RAL
from .basic_blocks_v2 import SetBlock, BasicConv2d

class SetNet3d(nn.Module):
    def __init__(self, hidden_dim,frame_num):
        super(SetNet3d, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        self.frame_num = frame_num
        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        
        self.layer_3d = SetBlock3d(BasicConv3d(_set_in_channels,_set_channels[0], [1,7,7], padding=[0,3,3]))
        
        self.M3D_1 = SetBlock3d(M3D_3(_set_channels[0], _set_channels[0]))
        self.layer1 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 5, padding=2))
        self.layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.RAL_1 = RAL(_set_channels[0], int(self.frame_num))
        
        self.M3D_2 = SetBlock3d(M3D_3(_set_channels[0], _set_channels[0]), True, kernel_size=[2,1,1], stride=[2,1,1])
        self.layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.RAL_2 = RAL(_set_channels[1], int(self.frame_num/2))
        
        self.M3D_3 = SetBlock3d(M3D_1(_set_channels[1], _set_channels[1]), True, kernel_size=[2,1,1], stride=[2,1,1])
        self.layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
        #self.RAL_3 = RAL(_set_channels[2], int(self.frame_num/4))
        
        #self.M3D_4 = SetBlock3d(M3D_1(_set_channels[2], _set_channels[2]), True, kernel_size=[2,1,1], stride=[2,1,1])
        
        self.Softmax = nn.Softmax() 
        self.Softmax2 = nn.Softmax(dim=2)
        self.bin_num = [1, 2, 4, 8, 16]

        self.fc_bin3d = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num), 320, hidden_dim)))])

        #各层参数的初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
    #set
    def frame_max(self, x):

        return torch.max(x, 2) #dim=1 也就是对30张frame_set做max操作 return a list contians 0:max data and 1:indices


    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list
    
    
    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        b, n, h, w = silho.size() # n=128
        

        n = silho.size(1) #frame_num

        x_3d = silho.view(-1,self.frame_num,h,w).unsqueeze(1)
      
        del silho

        x_3d = self.layer_3d(x_3d)  #[64,32,8,64,44]

        x_3d = self.M3D_1(x_3d) #(64,32,8,64,44)
        
        x_3d = self.layer1(x_3d)

        x_3d = self.layer2(x_3d)
        
        x_3d = self.RAL_1(x_3d)
        
        x_3d = self.M3D_2(x_3d)
        x_3d = self.layer3(x_3d)
        x_3d = self.layer4(x_3d)
        x_3d = self.RAL_2(x_3d)

        x_3d1 = self.M3D_3(x_3d)
        x_3d2 = self.layer5(x_3d1)
        x_3d3 = self.layer6(x_3d2)
    
        x_3d = torch.cat([x_3d1, x_3d2, x_3d3],1)

        #x_3d = self.RAL_3(x_3d)
        x_3d = self.frame_max(x_3d)[0]
        #x_3d = self.M3D_4(x_3d) #[64,32,8,64,44]

        x_3d = x_3d.squeeze(2)
        feature = list()

        n, c, h, w = x_3d.size()
        
        for num_bin in self.bin_num: #bin_num = [1,2,4,8,16]        
            z_3d = x_3d.view(n, c, num_bin, -1)
            z_3d = z_3d.mean(3) + z_3d.max(3)[0]
            feature.append(z_3d)
       
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()

        feature = feature.matmul(self.fc_bin3d[0])

        feature = feature.permute(1,0,2).contiguous()
        _, n, d = feature.size()
        feature = torch.mean(feature.view(b,-1, n, d),1)

        return feature, None