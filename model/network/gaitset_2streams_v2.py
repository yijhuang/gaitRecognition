import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from .basic_blocks import SetBlock, BasicConv2d

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
class SetNet(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        self.num_classes = num_classes
        _set_in_channels = 1
        _set_channels = [32, 64, 128]

        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
        
        self.bottleneck = nn.BatchNorm1d(hidden_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(hidden_dim, self.num_classes,bias=False)
        

        
        self.bin_num = [1, 2, 4, 8, 16]
        #self.bin_num = [1, 2, 4]

        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
        #            torch.zeros(sum(self.bin_num), 128, hidden_dim)))])
                    torch.zeros(sum(self.bin_num), 320, hidden_dim)))]) #nn.Parameter 将xxx转化成可训练的变量[62,128,256]

#initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias:
                    nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
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
    


    def forward(self, silho, batch_frame=None, training=True):
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


        feature = list()

        n, c, h, w = x.size()
        for num_bin in self.bin_num: #bin_num = [1,2,4,8,16]
            z = x.view(n, c, num_bin, -1)#维度变换num_bin=1,z.size=[]
            z = z.mean(3) + z.max(3)[0]#z.mean(3):[128,128,1];z.max(3)[0]:[128,128,1]
            feature.append(z)
            
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1,0,2).contiguous()
        feature_cls = feature.permute(0,2,1).contiguous()
        feature_cls = self.bottleneck(feature_cls).permute(0,2,1).contiguous()
        if training:
            cls_score = self.classifier(feature_cls)
            return feature, cls_score, None
        else:
            return feature, None
        
        #return feature, None
