import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from .network import TripletLoss, SetNet, SetNet3d, CrossEntropyLabelSmooth, CenterLoss
#from .network import TripletLoss,SetNet3d,SoftmaxLoss
from .utils import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 frame_num_3d,
                 model_name,
                 train_source,
                 test_source,
                 num_classes,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.frame_num_3d = frame_num_3d
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size
        self.num_classes = num_classes
        
        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        #self.encoder1 = SetNet(self.hidden_dim,self.frame_num).float()
        self.encoder1 = SetNet(self.hidden_dim, self.num_classes).float()
        self.encoder1 = nn.DataParallel(self.encoder1)
        self.encoder2 = SetNet3d(self.hidden_dim, self.frame_num_3d, self.num_classes).float()
        #self.encoder2 = SetNet3d(self.hidden_dim,self.frame_num_3d).float()
        self.encoder2 = nn.DataParallel(self.encoder2)
        
        self.center_criterion = CenterLoss(num_classes = num_classes)
        self.center_criterion3d = CenterLoss(num_classes = num_classes)
        self.center_criterion = nn.DataParallel(self.center_criterion)
        self.center_criterion3d = nn.DataParallel(self.center_criterion3d)
        
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.softmax_loss = CrossEntropyLabelSmooth(self.num_classes)
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.softmax_loss = nn.DataParallel(self.softmax_loss)
        
        self.encoder1.cuda()
        self.encoder2.cuda()
        self.center_criterion.cuda()
        self.center_criterion3d.cuda()

        
        self.triplet_loss.cuda()
        self.softmax_loss.cuda()
#         self.softmax_loss = SoftmaxLoss(self.P*self.M).float()
#         self.softmax_loss = nn.DataParallel(self.softmax_loss)
#         self.softmax_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder1.parameters()},
            {'params': self.encoder2.parameters()},
            {'params': self.center_criterion.parameters()},
            {'params': self.center_criterion3d.parameters()},
        ], lr=self.lr)

        #self.optimizer = optim.Adam([
        #     {'params': self.encoder2.parameters()}
        # ], lr=self.lr)
        
        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01
        
        self.sample_type1 = 'all'
        self.sample_type2 = 'all'
        self.training = True
    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]
        
        def select_frame(index,sample_type):

            sample = seqs[index]
            #frame_set[0,1,2,...,]
            frame_set = frame_sets[index]
            if sample_type == 'random':
                #print(frame_set)
                #input()
                frame_id_list = np.random.choice(frame_set, size=self.frame_num)      
                frame_id_list = frame_id_list.tolist()
                _ = [feature.loc[frame_id_list].values for feature in sample]
            elif sample_type == 'ordered':
                if len(frame_set) >=self.frame_num_3d:
                    frame_set_begin_list = frame_set[0:-self.frame_num_3d+1]
                    frame_id_begin = np.random.choice(frame_set_begin_list)
                    frame_id_list = [i for i in range(frame_id_begin,frame_id_begin+self.frame_num_3d)]
                    #frame_id_list = frame_id_list.tolist()
                    #frame_id_list = np.sort(frame_id_list)
                    _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    length = math.ceil(self.frame_num_3d/len(frame_set))
                    frame_set = frame_set*length
                    frame_id_list = frame_set[0:self.frame_num_3d]                   
                    _ = [feature.loc[frame_id_list].values for feature in sample]
            elif sample_type == 'all_ordered':
                if len(frame_set) >self.frame_num_3d:
                    if len(frame_set)%self.frame_num_3d != 0:
                        frame_id_list = frame_set[0:-(len(frame_set)%self.frame_num_3d)]
              
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                    else:
                        _ = [feature.values for feature in sample]
                else:
                    length = math.ceil(self.frame_num_3d/len(frame_set))
                    frame_set = frame_set*length
                    frame_id_list = frame_set[0:self.frame_num_3d]
                    _ = [feature.loc[frame_id_list].values for feature in sample]

            else:
                _ = [feature.values for feature in sample]                   
            return _
        
        seqs_3d = list(map(select_frame, range(len(seqs)), [self.sample_type2]*len(seqs)))
        seqs = list(map(select_frame, range(len(seqs)), [self.sample_type1]*len(seqs)))

        if self.sample_type1 == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        elif self.sample_type1 == 'ordered':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)
        
        if self.sample_type2 == 'random':
            seqs_3d = [np.asarray([seqs_3d[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        elif self.sample_type2 == 'ordered':
            seqs_3d = [np.asarray([seqs_3d[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        elif self.sample_type2 == 'all_ordered':
            seqs_3d = [np.asarray([seqs_3d[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs_3d = [[
                        np.concatenate([
                                           seqs_3d[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            #print(seqs_3d[0][0].shape)
            #input()
            seqs_3d = [np.asarray([
                                   np.pad(seqs_3d[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)
        
        batch[0] = seqs
        batch.append(seqs_3d)

        return batch
    
    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)
        
        self.encoder1.train()
        self.encoder2.train()
        self.center_criterion.train()
        self.center_criterion3d.train()
        
        self.sample_type1 = 'random'
        self.sample_type2 = 'ordered'
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        #collate_fn1 = self.collate_fn(sample_type = self.sample_type1)
        #collate_fn2 = self.collate_fn(sample_type = self.sample_type2)
        #print(collate_fn1,collate_fn2)
        #input()

        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)
        #self.sample_type = 'ordered'
#         train_loader2 = tordata.DataLoader(
#             dataset=self.train_source,
#             batch_sampler=triplet_sampler,
#             collate_fn=self.collate_fn2,
#             num_workers=self.num_workers)
#         for loader1,loader2 in zip(train_loader1, train_loader2):
#             seq1, view1, seq_type1, label1, batch_frame1 = loader1[0],loader1[1],loader1[2],loader1[3],loader1[4]
#             seq2, view2, seq_type2, label2, batch_frame2 = loader2[0],loader2[1],loader2[2],loader2[3],loader2[4]
#             print(seq1[0].shape,seq2[0].shape)
#             input()
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        
        #for loader1,loader2 in zip(train_loader1,train_loader2):
        for seq, view, seq_type, label, batch_frame, seq_3d in train_loader:
            #seq1, view1, seq_type1, label1, batch_frame1 = loader1[0],loader1[1],loader1[2],loader1[3],loader1[4]
            #seq2, view2, seq_type2, label2, batch_frame2 = loader2[0],loader2[1],loader2[2],loader2[3],loader2[4]
            self.restore_iter += 1
            self.optimizer.zero_grad()
            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
                seq_3d[i] = self.np2var(seq_3d[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            #feature, feature_softmax, label_prob = self.encoder(*seq, batch_frame)

            #len(seq[0]) = batchsize
            #seq[0][0].shape = [30,64,44]
            #seq_3d[0][0].shape = [8,64,44]
            feature, cls_score, label_prob = self.encoder1(*seq, batch_frame, self.training)
            #feature3d, label_prob3d = self.encoder2(*seq_3d, batch_frame)
            feature3d, cls_score3d,label_prob3d = self.encoder2(*seq_3d, batch_frame, self.training)

            ctr_feature = feature.permute(1, 0, 2).contiguous()
            ctr_feature3d = feature3d.permute(1,0,2).contiguous()
            
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()
            
            ctr_label = target_label.unsqueeze(0).repeat(ctr_feature.size(0), 1)
            ctr_label3d = target_label.unsqueeze(0).repeat(ctr_feature3d.size(0), 1)
            
            
            feature = torch.cat([feature,feature3d],1) 
            #feature = feature3d
            triplet_feature = feature.permute(1, 0, 2).contiguous()
 
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            #label_softmax = target_label.unsqueeze(1).repeat(1,feature_softmax.size(1))

            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)

            sfx_feature = cls_score.permute(1, 0, 2).contiguous()
            sfx_feature3d = cls_score3d.permute(1, 0, 2).contiguous()
            sfx_label = target_label.unsqueeze(0).repeat(sfx_feature.size(0), 1)
            sfx_label3d = target_label.unsqueeze(0).repeat(sfx_feature3d.size(0), 1)
            
            softmax_loss = self.softmax_loss(sfx_feature, sfx_label).mean()
            softmax_loss3d = self.softmax_loss(sfx_feature3d, sfx_label3d).mean()
            center_loss = self.center_criterion(ctr_feature, ctr_label)
            center_loss3d = self.center_criterion3d(ctr_feature3d, ctr_label3d)
            
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()
            
            #loss = loss + softmax_loss

            loss = loss + softmax_loss + 0.0005 * center_loss + softmax_loss3d + 0.0005 * center_loss3d
            
            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 1000 == 0:
                self.save()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
                


            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        #self.encoder1.eval()
        self.encoder2.eval()
        source = self.test_source if flag == 'test' else self.train_source
        #self.sample_type1 = 'all'
        self.sample_type1 = 'all'
        self.sample_type2 = 'all_ordered'
        self.training = False
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        feature_3d_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        for i, x in enumerate(data_loader):
            
            seq, view, seq_type, label, batch_frame, seq_3d = x    #(seq:frameset,view:angle,seq,label,batch_frame:frame_num)

            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
                seq_3d[j] = self.np2var(seq_3d[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            #feature, feature_softmax, _ = self.encoder(*seq, batch_frame)
            feature, _ = self.encoder1(*seq, batch_frame, self.training)
            #feature_3d, _ = self.encoder2(*seq_3d, batch_frame)
            #feature = torch.nn.functional.normalize(feature, dim=2)
            feature_3d, _ = self.encoder2(*seq_3d, None, self.training)
            #feature_3d = torch.nn.functional.normalize(feature_3d, dim=2)
            #feature = feature_3d

            
            n, num_bin, _ = feature.size()
            feature = torch.nn.functional.normalize(feature.view(n,-1),dim=1)
            feature_3d = torch.nn.functional.normalize(feature_3d.view(n,-1), dim=1)

            #feature = torch.cat([feature,feature_3d],1)

            feature_list.append(feature.data.cpu().numpy())
            feature_3d_list.append(feature_3d.data.cpu().numpy())

            view_list += view
            seq_type_list += seq_type
            label_list += label

        #return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
        return np.concatenate(feature_list, 0), np.concatenate(feature_3d_list, 0), view_list, seq_type_list, label_list
    
    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        state = {'gaitset':self.encoder1.state_dict(),
                 'conv3dNet':self.encoder2.state_dict(),
                 'center_loss':self.center_criterion.state_dict(),
                 'center_loss3d': self.center_criterion3d.state_dict(),
                }

        torch.save(state,
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))
        
    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        checkpoint = torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter)))
        checkpoint3d = torch.load(osp.join(
            'checkpoint', 'M3D_RAL_cat_2sfxloss_2ctrloss_temporal_74',
            'M3D_RAL_cat_2sfxloss_2ctrloss_temporal_74_CASIA-B_73_False_256_0.2_64_full_30-80000-encoder.ptm'))
        self.encoder1.load_state_dict(checkpoint['gaitset'])
        self.encoder2.load_state_dict(checkpoint3d['conv3dNet'])
        self.center_criterion.load_state_dict(checkpoint['center_loss'])
        self.center_criterion3d.load_state_dict(checkpoint3d['center_loss'])
        #self.encoder2.load_state_dict(checkpoint['gaitset'])
        
        #self.optimizer.load_state_dict(torch.load(osp.join(
        #    'checkpoint', self.model_name,
        #    '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
