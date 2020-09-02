import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from .basic_blocks import SetBlock, BasicConv2d

class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]

        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
        
        self.bin_num = [1, 2, 4, 8, 16]
        #self.bin_num = [1, 2, 4]

        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
        #            torch.zeros(sum(self.bin_num), 128, hidden_dim)))])
                    torch.zeros(sum(self.bin_num), 320, hidden_dim)))]) #nn.Parameter 将xxx转化成可训练的变量[62,128,256]

        #self.fc_bin3 = nn.ParameterList([
        #    nn.Parameter(
        #        nn.init.xavier_uniform_(
        #            torch.zeros( 2*sum(self.bin_num), hidden_dim, 124)))])
        #self.fc_para = nn.ParameterList([
        #    nn.Parameter(
        #        nn.init.xavier_uniform_(
        #            torch.zeros(1,2)))])
        
        #各层参数的初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)
    #set
    def frame_max(self, x):
        #if self.batch_frame is None:
        return torch.max(x, 2) #dim=1 也就是对30张frame_set做max操作 return a list contians 0:max data and 1:indices
        
        #else:
        #    #batch_frame = the number of frames
        #    print('x:',x.shape)
        #    input()
        #    _tmp = [
        #        torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
        #        for i in range(len(self.batch_frame) - 1)
        #        ] #求batch_frame中的最大值
        #    max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
        #    arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
        #    print('max_list:',max_list.shape)
        #    input()
        #    return max_list, arg_max_list

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
    
    #def hpm(self,x):
    #    feature = list()
    #    #gl.size = [128,128,16,11]
    #    n, c, h, w = x.size()
    #    
    #    for num_bin in self.bin_num: #bin_num = [1,2,4,8,16]
    #        z = x.view(n, c, num_bin, -1)#维度变换num_bin=1,z.size=[]
    #        z = z.mean(3) + z.max(3)[0]#z.mean(3):[128,128,1];z.max(3)[0]:[128,128,1]
    #        feature.append(z)
    #        #z = gl.view(n, c, num_bin, -1)#同上处理
    #        #z = z.mean(3) + z.max(3)[0]#z.max(3):dim=1时为索引，z.shape=[128,128,1]
    #        #feature type:list length=10
    #        #feature.append(z) 
    #    #feature shape before permute:[128,128,62],feature after permute:[62,128,128]
    #    feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
    #    return feature

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
        n = silho.size(0) # n=128
        x = silho.unsqueeze(2)
        del silho

        x_1 = self.set_layer1(x)
        x_2 = self.set_layer2(x_1)
        x_3 = self.set_layer3(x_2)
        x_4 = self.set_layer4(x_3)
        x_5 = self.set_layer5(x_4)
        x_6 = self.set_layer6(x_5)
        x = torch.cat([x_4,x_5,x_6],2)
        x = torch.max(x,1)[0]

       #parameters 
        #x = 0.5 * x + 0.5 * x_3d
        #x.shape = [128,30,64,32,22]
        # feature = self.hpm(x)
        #x = torch.cat([x,x_3d],1)
        feature = list()
        #gl.size = [128,128,16,11]
        n, c, h, w = x.size()
        for num_bin in self.bin_num: #bin_num = [1,2,4,8,16]
            z = x.view(n, c, num_bin, -1)#维度变换num_bin=1,z.size=[]
            z = z.mean(3) + z.max(3)[0]#z.mean(3):[128,128,1];z.max(3)[0]:[128,128,1]
            feature.append(z)
            
            #z = gl.view(n, c, num_bin, -1)#同上处理
            #z = z.mean(3) + z.max(3)[0]#z.max(3):dim=1时为索引，z.shape=[128,128,1]
            #feature type:list length=10
            #feature.append(z) 
        # #feature shape before permute:[128,128,62],feature after permute:[62,128,128]
        #feature.shape = [62,128,256]
        #feature.shape = [128,62,256] [batchsize, bins, out_dim]
        #feature = 0.75*feature + 0.25*feature_3d
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])
        #feature_softmax = feature.matmul(self.fc_bin3[0])
        feature = feature.permute(1,0,2).contiguous()
        #feature_softmax = feature_softmax.permute(1,0,2).contiguous() 
        #return feature, feature_softmax, None
        return feature, None
