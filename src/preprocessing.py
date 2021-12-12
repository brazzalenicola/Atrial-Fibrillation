from scipy.io import loadmat, savemat
import pandas as pd
import os
import torch
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset
mpl.rcParams['figure.figsize'] = (16,10)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = 'large'


def load_data(file_name, data_dir):
    #ref = pd.read_csv('data/REFERENCE-v3.csv', names=['mat', 'label'], index_col=0)
    # Load the ECG signal from the .mat file
    file_mat = data_dir.decode() + '/' + file_name.decode() + '.mat'
    data = loadmat(file_mat)['val']
    #labs = ref["label"]
    #lab = labs.loc[file_name.decode()]
    return data.squeeze()#, lab

def plot_ecg(data, file_name, lbl):
    dic = {'N': "Normal",
        'A' : "Atrial Fibrillation",
        'O': "Abnormal ryhthm",
        '~' : "Noisy"}
    plt.figure()
    plt.plot(data, color='b')
    plt.xlim([4000, 7000]) 
    title = str(dic[lbl]) + " " + file_name.decode()
    plt.title('{} ECG'.format(title))

def baseline_wander_removal(data):
    # Sampling frequency
    fs = 300
      
    win_size = int(np.round(0.2 * fs)) + 1
    baseline = medfilt(data, win_size)
      
    win_size = int(np.round(0.6 * fs)) + 1 
    baseline = medfilt(baseline, win_size)

    # Removing baseline
    filt_data = data - baseline
    return filt_data

def normalize_data(data):
    # Amplitude estimate
    norm_factor = np.percentile(data, 99) - np.percentile(data, 5)
    return (data / norm_factor)


def random_crop(data, target_size=9000, center_crop=False):
    
    N = data.shape[0]
    # Return data if correct size
    if N == target_size:
        return data
    
    # If data is too small, then pad with zeros
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        return np.pad(data, [left_pads, right_pads], mode='constant')
    
    # Random Crop (always centered if center_crop=True)
    if center_crop:
        # set the starting point to perform a crop in the middle
        from_ = int((N / 2) - (target_size / 2))
    else:
        from_ = np.random.randint(0, np.floor(N - target_size))
    to_ = from_ + target_size
    return data[from_:to_]


def load_and_preprocess_data(file_name, data_dir):
    # Load data
    data = load_data(file_name, data_dir)
    # Baseline wander removal
    data = baseline_wander_removal(data)
    # Normalize
    data = normalize_data(data)
    # Random Crop 
    data = random_crop(data, center_crop=True)

    return data.astype(np.float32)
    


"""Loading the noisy samples"""

class PhysioNetDataset(Dataset):
    def __init__(self, ref, data_dir):
        self.data_dir = data_dir
        self.ref_frame = ref

    def __len__(self):
        return len(self.ref_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
          
        file_name = self.ref_frame.index[idx].encode()

        data = load_and_preprocess_data(file_name, self.data_dir)

        sample = {'ecg': data, 'label': self.ref_frame.iloc[idx,1]}

        return sample

#creates a dataset containing only the under represented class
class ImbalancedDataset(Dataset): 
    def __init__(self, ref, data_dir, label_code):
        self.data_dir = data_dir
        self.ref_frame = ref[ref['label_code'] == label_code]

    def __len__(self):
        return len(self.ref_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
          
        file_name = self.ref_frame.index[idx].encode()

        data = load_imbalance_data(file_name, self.data_dir)

        sample = {'ecg': data, 'label': self.ref_frame.iloc[idx,1]}

        return sample