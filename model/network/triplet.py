import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).byte().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).byte().view(-1)
        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        
        full_loss_metric_mean[full_loss_num == 0] = 0
        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        #self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).scatter_(2, targets.unsqueeze(2).data.cpu(), 1).cuda()

        #if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).mean(0).sum()

        return loss

class CenterLoss(nn.Module):
    """Center Loss.
    Args:
        num_classes(int): number of classes.
        feat_dim(int): feature dimension.
    """
    def __init__(self, num_classes=124, feat_dim = 256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.centers = nn.ParameterList([nn.Parameter(torch.randn(31, self.num_classes, self.feat_dim))])
        #self.centers = nn.Parameter(torch.randn(31, self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels ground truth labels with shape (num_classes).
        """

        assert x.size(0) == labels.size(0),"features.size(0) is not equal to labels.size(0)"

        n,batch_size,_ = x.size()
        distmat = torch.pow(x, 2).sum(dim=2, keepdim=True).expand(n, batch_size, self.num_classes) + \
               torch.pow(self.centers[0], 2).sum(dim=2, keepdim=True).expand(n, self.num_classes, batch_size).permute(0,2,1).contiguous()

        # distmat.shape:[31, 64, 124]

        #distmat.addmm_(1, -2, x, self.centers.transpose(1,2).contiguous())
        #shape: [31, 64, 124]
        distmat = distmat + (-2) * x.matmul(self.centers[0].transpose(1,2).contiguous())

        classes = torch.arange(self.num_classes).long()

        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(2).expand(n, batch_size, self.num_classes)

        mask = labels.eq(classes.expand(n, batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / (batch_size * n) #clamp 限制范围

        return loss
        