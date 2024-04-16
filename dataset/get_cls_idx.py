import os
import numpy as np
path_data_root = '/mnt/guided/plan2data/corpus-time-my'
train_data_name = 'train_data_linear'
path_train_data = os.path.join(path_data_root,  train_data_name + '.npz')
os.path.exists(path_train_data)

train_data = np.load(path_train_data)
train_x = train_data['x']

class_1_idx = []
class_2_idx = []
class_3_idx = []
class_4_idx = []
idxs = [class_1_idx, class_2_idx, class_3_idx, class_4_idx]
for i, sample in enumerate(train_x):
    idxs[sample[0][-1] - 1].append(i)

class_1_idx[:10]

cls_1 = train_x[class_1_idx]
cls_1[0][0][-1]

path_train = os.path.join(path_data_root, train_data_name + '_data_idx.npz')
np.savez(
        path_train, 
        cls_1_idx=class_1_idx,
        cls_2_idx=class_2_idx,
        cls_3_idx=class_3_idx,
        cls_4_idx=class_4_idx
    )
