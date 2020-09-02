import torch.utils.data as tordata
import random 
import numpy as np


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = []
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0])
            #print(pid_list)
            #input()
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = np.random.choice(
                    _index,
                    size=self.batch_size[1])
                _index = _index.tolist()
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
