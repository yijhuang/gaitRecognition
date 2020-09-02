import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import random
import math
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution, is_train=True):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.is_train = is_train
        self.label_set = set(self.label)
        #print(self.label_set)
        #input()
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])
        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path, is_train):
        if is_train:
            return self.randomErasing(self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0)
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32')
    
    
    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path, self.is_train) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path, self.is_train) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict
    
    def randomErasing(self, img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914,0.4822,0.4465)):

        if random.uniform(0,1) >= probability:
            return img
        for attempt in range(100):
 
            area = img.shape[1] * img.shape[2]
            
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            
            if w < img.shape[2] and h< img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                img[0, x1:x1 + h,y1:y1 + w] = mean[0]
            return img
        
        return img
             
    def __len__(self):
        return len(self.label)
