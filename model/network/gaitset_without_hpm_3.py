import torch
import torch.nn as nn
import numpy as np

from .basic_blocks import SetBlock, BasicConv2d


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128, 960, 320]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))#c1
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)#c2+pool
        #self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), kernel_size=2)#c2+pool
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))#c3
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)#c4+pool
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))#c5
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))#c6+pool
        
        self.non_block = SetBlock(BasicConv2d(_set_channels[3], _set_channels[4], 1))

        #self.sampling1 = SetBlock(nn.AvgPool2d(2,2))
        #self.sampling2 = SetBlock(nn.AvgPool2d(4,4))

        #self.sub1_layer = SetBlock(BasicConv2d(_set_in_channels, 96, 8, padding=4),
        #    True, kernel_size=7, stride=2, padding=2)
        #self.sub2_layer = SetBlock(BasicConv2d(_set_in_channels, 96, 5, padding=2))


        # _gl_in_channels = 32
        # _gl_channels = [64, 128]
        # self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)#c3
        # self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)#c4
        # self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)#c5
        # self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)#c6
        # self.gl_pooling = nn.MaxPool2d(2)

        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num), 320, hidden_dim)))]) #nn.Parameter 将xxx转化成可训练的变量[62,128,256]
       # self.fc_bin2 = nn.ParameterList([
       #     nn.Parameter(
       #         nn.init.xavier_uniform_(
       #             torch.zeros(sum(self.bin_num), 96, hidden_dim)))])

       # self.fc_bin3 = nn.ParameterList([
       #     nn.Parameter(
       #         nn.init.xavier_uniform_(
       #             torch.zeros(sum(self.bin_num), 96, hidden_dim)))])

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
        if self.batch_frame is None:
            return torch.max(x, 1) #dim=1 也就是对30张frame_set做max操作 return a list contians 0:max data and 1:indices
        
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ] #求batch_frame中的最大值
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_mean(self, x):
        if self.batch_frame is None:
            return torch.mean(x, 1) #dim=1 也就是对30张frame_set做max操作 return a list contians 0:max data and 1:indices
        
        else:
            _tmp = [
                torch.mean(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ] #求batch_frame中的最大值
            mean_list = torch.cat([_tmp[i] for i in range(len(_tmp))], 0)
            #arg_mean_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            #return mean_list, arg_mean_list
            return mean_list

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

    #    n, c, h, w = x.size()
    #    for num_bin in self.bin_num:
    #        z = x.view(n, c, num_bin, -1)
    #        z = z.mean(3) + z.max(3)[0]
    #        feature.append(z)
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
        x = silho.unsqueeze(2) #在siloho的dim=2增加一个维度=1 x.shape = [128,30,1,64,44] channels = 1
        #x_sub1 = self.sampling1(x)
        #x_sub2 = self.sampling2(x)
        del silho
        #x.shape : [128,30,32,64,44][batchsize,frame_num,channels,w,h]
        x1 = self.set_layer1(x)
        #x.shape : [128,30,32,32,22][batchsize,frame_num,channels,w,h]
        x2 = self.set_layer2(x1)
        #self.frame_max(x)[0].shape = [128,32,32,22] gl.shape = [128,64,32,22]
        #gl = self.gl_layer1(self.frame_max(x)[0]) #self.frame_max[0]is data,self.frame_max[1] is indices
        #gl.shape = [128,64,32,22]
        #gl = self.gl_layer2(gl)
        #gl.shape = [128,64,16,11]
        #gl = self.gl_pooling(gl)
        #x.shape = [128,30,64,32,22]
        x3 = self.set_layer3(x2)
        #x.shape = [128,30,64,16,11]
        x4 = self.set_layer4(x3)
        #gl.shape = [128,128,16,11]
        #gl = self.gl_layer3(gl + self.frame_max(x)[0])
        #gl.shape = [128,128,16,11]
        #gl = self.gl_layer4(gl)
        #x.shape = [128,30,128,16,11]
        x5 = self.set_layer5(x4)
        #x.shape = [128,30,128,16,11]
        x6 = self.set_layer6(x5)
        #x.shape = [128,128,16,11]
        x = torch.cat([x4,x5,x6],2)
        #x = self.frame_max(x)[0] + self.frame_mean(x)[0]
        _, frame_num, _, _, _ = x.shape

        x_max = self.frame_max(x)[0].unsqueeze(1).repeat(1,frame_num,1,1,1)

        x_mean = self.frame_mean(x).unsqueeze(1).repeat(1,frame_num,1,1,1)
        x_median = self.frame_median(x)[0].unsqueeze(1).repeat(1,frame_num,1,1,1)
        x_cat = torch.cat([x_max,x_mean,x_median],2)
        x = self.non_block(x_cat)*x + x
        x = self.frame_max(x)[0] + self.frame_mean(x)
        
        #gl = gl + x
       # x_sub1 = self.sub1_layer(x_sub1)
       # x_sub1 = self.frame_max(x_sub1)[0]
       # x_sub2 = self.sub2_layer(x_sub2)
       # x_sub2 = self.frame_max(x_sub2)[0]
        #feature = self.hpm(x)
        #feature2 = self.hpm(x_sub1)
        #feature3 = self.hpm(x_sub2)
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
         #feature shape before permute:[128,128,62],feature after permute:[62,128,128]
       
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
       #feature.shape = [62,128,256]
        feature = feature.matmul(self.fc_bin[0]) #（只是矩阵乘法）
       #feature2 = feature2.matmul(self.fc_bin2[0]) #（只是矩阵乘法）
        #feature3 = feature3.matmul(self.fc_bin3[0]) #（只是矩阵乘法）
        
        #feature = torch.cat([feature1,feature2,feature3],2)
        #feature.shape = [128,62,256] [batchsize, bins, out_dim]
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None
