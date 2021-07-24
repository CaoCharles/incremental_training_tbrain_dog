import os
import pandas as pd
import numpy as np

import random
import librosa

from torch.utils.data import Dataset

# try:
#     def wav_read(wav_file):
#         wav_data, sr = sf.read(wav_file, dtype='int16')
#         return wav_data, sr

# except ImportError:
#     def wav_read(wav_file):
#         raise NotImplementedError('WAV file reading requires soundfile package.')

# 製作頻譜圖
def get_melspec(prefix, file_name):
  y, sr = librosa.load(f'{prefix}/{file_name}.wav', sr = 8000)
  if len(y)<sr*5:
      print(prefix,file_name)
      l = sr*5-len(y)
      y = np.hstack([y,np.zeros(l)])
  mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=128, n_fft=1024, hop_length=256))
  return mel_spec

# 資料增強(SpecAugment)
def spec_augment(spec: np.ndarray, num_mask=3, 
                 freq_masking_max_percentage=0.1, time_masking_max_percentage=0.1):
    '''Crate Augment Data'''
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def add_noise(spec):
    h, w = spec.shape
    amp = np.random.uniform(0,spec.max()/10)
    white_noise = np.random.randn(h, w)*amp
    return spec+white_noise

def time_shift(spec):
    cut = np.random.choice(spec.shape[1])
    return np.hstack([spec[:,cut:],spec[:,:cut]])

def add_gain(spec):
    gain = np.random.choice(25)-12
    return spec+spec

# 讀取資料
class SoundDataset(Dataset):
    """Create Sound Dataset with loading wav files and labels from csv.

    Attributes:
        params: A class containing all the parameters.
        data_type: A string indicating train or val.
        csvfile: A string containing our wav files and labels.
        mixup: A boolean indicating whether to do mixup augmentation or not.
    """
    def __init__(self, spec, label=None, mode='train'):
        """Init SoundDataset with params
        Args:
            params (class): all arguments parsed from argparse
            train (bool): train or val dataset
        """
        # 主講者寫的
#         self.params = params
#         self.csvfile = params.csv_path
#         self.data_dir = params.data_dir

        # 我們需要的
        self.specs = spec
        self.labels = label
        self.mode = mode

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Augment here if you want
        mel_spec = self.specs[idx]
        # Stack mel_spec data 
        if self.mode=='train':
            p = 0.8
            if random.random()<p: mel_spec = add_noise(mel_spec)
            if random.random()<p: mel_spec = time_shift(mel_spec)
            if random.random()<p: mel_spec = add_gain(mel_spec)
            if random.random()<p:
                mel_spec = spec_augment(mel_spec,num_mask=3, freq_masking_max_percentage=0.1, time_masking_max_percentage=0.1)
        
        # Stack 組合成照片
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))/255
        # print(mel_spec.shape)
        return mel_spec, self.labels[idx]


# class SoundDataset_test(SoundDataset):
#     def __init__(self, params):
#         self.params = params
#         self.csvfile = params.csv_path
#         self.data_dir = params.data_dir
#         self.normalize = self.params.normalize
#         self.preload = self.params.preload

#         self.X, self.Y, self.filenames = self.read_data(self.csvfile)
#         if self.preload:
#             self.X = self.convert_to_spec(self.X)
#             self.shape = self.get_shape(self.X[0])
#         else:
#             self.shape = self.get_shape(self.preprocessing(self.X[0][0], self.X[0][1]))

#     def read_data(self, csvfile):
#         df = pd.read_csv(csvfile)
#         data, label, filenames = [], [], []
#         print("reading wav files...")
#         for i in tqdm(range(len(df))):
#             row = df.iloc[i]
#             path = os.path.join(self.data_dir, row.Filename + ".wav")
#             wav_data, sr = wav_read(path)
#             assert wav_data.dtype == np.int16
#             data.append((wav_data, sr))
#             lb = None
#             if row.Barking == 1:
#                 lb = 0
#             elif row.Howling == 1:
#                 lb = 1
#             elif row.Crying == 1:
#                 lb = 2
#             elif row.COSmoke == 1:
#                 lb = 3
#             elif row.GlassBreaking == 1:
#                 lb = 4
#             elif row.Other == 1:
#                 lb = 5
#             label.append(lb)
#             filenames.append(path)
#         return data, label, filenames