import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import glob
from numpy.random import default_rng

def MR_NIRP_split(val_num=1):

    h5_dir = '../datasets/MR-NIRP'
    train_list = []
    val_list = []

    for subject in range(1,9):
        for task in ['motion', 'still']:
            if os.path.isfile(h5_dir+'/sub%d_%s.h5'%(subject, task)):
                if subject == val_num:
                    val_list.append(h5_dir+'/sub%d_%s.h5'%(subject, task))
                else:
                    train_list.append(h5_dir+'/sub%d_%s.h5'%(subject, task))

    return train_list, val_list
    
def OBF_split(k=10, idx=0):
    h5_dir = '../OBF/h5_align'
    if idx>=k:
        raise(ValueError('invalid idx'))

    train_list = []
    val_list = []

    val_len = 100//k
    val_subject = list(range(idx*val_len+1, (idx+1)*val_len+1))

    for subject in range(1,101):
        for sess in [1,2]:
            if os.path.isfile(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess)):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess))
    return train_list, val_list  

def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = '../datasets/UBFC_h5'
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1,50):
        if os.path.isfile(h5_dir+'/%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%d.h5'%(subject))

    return train_list, val_list    


def MMSE_split_percentage(k=5, idx=0):

    f_sub_num = list(range(5,20)) + list(range(21,28))
    m_sub_num = list(range(1, 18))

    sub = np.array(['F%03d'%n for n in f_sub_num]+['M%03d'%n for n in m_sub_num])

    rng = np.random.default_rng(12345)
    sub = rng.permutation(sub)

    val_len = len(sub)//k
    sub_val = sub[idx*val_len+1:(idx+1)*val_len+1]
    
    all_files_list = glob.glob('../datasets/MMSE_HR_h5/*h5')

    train_list = []
    val_list = []

    for f_name in all_files_list:
        sub = f_name.split('/')[-1][:4]
        if sub in sub_val:
            val_list.append(f_name)
        else:
            train_list.append(f_name)

    return train_list, val_list
    
def PURE_split():

    h5_dir = '../datasets/PURE_h5'
    train_list = []
    val_list = []

    val_subject = [6, 8, 9, 10]

    for subject in range(1,11):
        for sess in [1,2,3,4,5,6]:
            if os.path.isfile(h5_dir+'/%02d-%02d.h5'%(subject, sess)):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
    return train_list, val_list  
    
class H5Dataset_(Dataset):

    def __init__(self, train_list, T, label_ratio):
        # TODO: Please note that the following code in __init__ is for fully labeled dataset. We manually set label_ratio to control how many labels are used for training.
        # TODO: if some videos have labels while others not in your dataset, please make sure the the labeled videos are all at the front of the self.train_list, and self.label_sample_number is the number of labeled videos.
        # TODO: Please modify self.train_list and self.label_sample_number according to your dataset.
        self.train_list = np.random.permutation(train_list) # list of .h5 file paths for training
        self.T = T # video clip length
        self.label_sample_number = int(len(self.train_list) * label_ratio) # number of samples with labels
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if idx < self.label_sample_number:
            label_flag = np.float32(1) # a flag to indicate whether the sample has a label
        else:
            label_flag = np.float32(0)
        
        h5_f = np.random.choice(glob.glob(self.train_list[idx]+'/*.h5'))

        with h5py.File(h5_f, 'r') as f:
            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq, bvp, label_flag

class H5Dataset(Dataset):

    def __init__(self, train_list, T, label_ratio):
        self.train_list = np.random.permutation(train_list) # list of .h5 file paths for training
        self.T = T # video clip length
        self.label_sample_number = int(len(self.train_list) * label_ratio) # number of samples with labels

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if idx < self.label_sample_number:
            label_flag = np.float32(1) # a flag to indicate whether the sample has a label
        else:
            label_flag = np.float32(0)

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq, bvp, label_flag

if __name__ == "__main__":
    a, b = MMSE_split_percentage(0.4)