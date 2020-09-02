import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
#    x_3d = torch.from_numpy(x_3d).cuda()
    y = torch.from_numpy(y).cuda()
#    y_3d = torch.from_numpy(y_3d).cuda()

#     x_y = x.matmul(y.transpose(0,1))
#     x_norm = torch.sqrt(torch.sum(x ** 2, 1)).unsqueeze(1)
#     y_norm = torch.sqrt(torch.sum(y ** 2, 1)).unsqueeze(0)
#     dist = 1 - x_y/x_norm.matmul(y_norm)

#     x_y_3d = x_3d.matmul(y_3d.transpose(0,1))
#     x_3d_norm = torch.sqrt(torch.sum(x_3d ** 2,1)).unsqueeze(1)
#     y_3d_norm = torch.sqrt(torch.sum(y_3d ** 2,1)).unsqueeze(0)
#     dist_3d = 1 - x_y_3d/x_3d_norm.matmul(y_3d_norm)

    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
         1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
#     dist_3d = torch.sum(x_3d ** 2, 1).unsqueeze(1) + torch.sum(y_3d ** 2, 1).unsqueeze(
#         1).transpose(0, 1) - 2 *torch.matmul(x_3d, y_3d.transpose(0, 1))
#     dist_3d = torch.sqrt(F.relu(dist))
#     print(dist)
#     print(dist_3d)
#     input()

    return dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    #feature, view, seq_type, label = data
    feature, feature_3d, view, seq_type, label = data

    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)
    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    #gallery_x.shape=[248,7936] gallery_x_3d.shape=[248,7936]
                    gallery_3d_x = feature_3d[gseq_mask, :]
                    gallery_y = label[gseq_mask]
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_3d_x = feature_3d[pseq_mask,:]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    dist_3d = cuda_dist(probe_3d_x, gallery_3d_x)

                    dist = 0.85 * dist + 0.15* dist_3d
                    
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc
