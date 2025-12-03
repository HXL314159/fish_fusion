import librosa
import glob
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from scipy.signal import resample
from itertools import chain
import time
import torchaudio
import random


def load_audio(path, sr):
    y, _ = librosa.load(path, sr)
    y = resample(y, num=sr*2)
    return y


def get_wav_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    path = '/vol/research/Fish_tracking_master/Fish_av_dataset/audio_dataset'
    audio = []
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path, dir))
        for dir1 in l2:
            wav_dir = os.path.join(path, dir, dir1, split, '*.wav')
            audio.append(glob.glob(wav_dir))
    return list(chain.from_iterable(audio))


def awgn(audio, snr):
    audio_power = audio**2
    audio_average_power = np.mean(audio_power)
    audio_average_db = 10*np.log10(audio_average_power)
    noise_average_db = audio_average_db - snr
    noise_average_power = 10**(noise_average_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    return audio + noise


def data_generator(seed, test_sample_per_class):
    """
    class to label mapping:
    none: 0
    strong: 1
    middle: 2
    weak: 3
    """

    random_state = np.random.RandomState(seed)
    strong_list = get_wav_name(split='strong')
    medium_list = get_wav_name(split='medium')
    weak_list = get_wav_name(split='weak')
    none_list = get_wav_name(split='none')

    random_state.shuffle(strong_list)
    random_state.shuffle(medium_list)
    random_state.shuffle(weak_list)
    random_state.shuffle(none_list)

    strong_test = strong_list[:test_sample_per_class]
    medium_test = medium_list[:test_sample_per_class]
    weak_test = weak_list[:test_sample_per_class]
    none_test = none_list[:test_sample_per_class]

    strong_val = strong_list[test_sample_per_class:2*test_sample_per_class]
    medium_val = medium_list[test_sample_per_class:2*test_sample_per_class]
    weak_val = weak_list[test_sample_per_class:2*test_sample_per_class]
    none_val = none_list[test_sample_per_class:2*test_sample_per_class]

    strong_train = strong_list[2*test_sample_per_class:]
    medium_train = medium_list[2*test_sample_per_class:]
    weak_train = weak_list[2*test_sample_per_class:]
    none_train = none_list[2*test_sample_per_class:]

    train_dict = []
    test_dict = []
    val_dict = []

    for wav in strong_train:
        train_dict.append([wav, 1])
    
    for wav in medium_train:
        train_dict.append([wav, 2])
    
    for wav in weak_train:
        train_dict.append([wav, 3])

    for wav in none_train:
        train_dict.append([wav, 0])
    
    for wav in strong_test:
        test_dict.append([wav, 1])
    
    for wav in medium_test:
        test_dict.append([wav, 2])
    
    for wav in weak_test:
        test_dict.append([wav, 3])

    for wav in none_test:
        test_dict.append([wav, 0])

    for wav in strong_val:
        val_dict.append([wav, 1])

    for wav in medium_val:
        val_dict.append([wav, 2])

    for wav in weak_val:
        val_dict.append([wav, 3])

    for wav in none_val:
        val_dict.append([wav, 0])

    random_state.shuffle(train_dict)

    return train_dict, test_dict, val_dict


class Fish_Voice_Dataset(Dataset):
    def __init__(self, sample_rate, seed, split):
        """
        split: train or test
        if sample_rate=None, read audio with the default sr
        """
        self.seed = seed
        self.split = split
        train_dict, test_dict, val_dict = data_generator(self.seed, test_sample_per_class=700)
        if self.split == 'train':
            self.data_dict = train_dict
        elif self.split == 'test':
            self.data_dict = test_dict
        elif self.split == 'val':
            self.data_dict = val_dict
        self.sample_rate = sample_rate

    def __len__(self):

        return len(self.data_dict)
    
    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio(wav_name, sr=self.sample_rate)
        wav = np.array(wav)
        # change 'eye(num)' if using different class nums
        target = np.eye(4)[target]

        data_dict = {'audio_name': wav_name, 'waveform': wav, 'target': target}

        return data_dict


def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    # wav = torch.stack([data['waveform'] for data in batch])
    target = [data['target'] for data in batch]
    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))
    # return wav, target
    return {'audio_name': wav_name, 'waveform': wav, 'target': target}


def get_dataloader(split,
                   batch_size,
                   sample_rate,
                   seed,
                   shuffle=False,
                   drop_last=False,
                   num_workers=8):

    dataset = Fish_Voice_Dataset(split=split, sample_rate=sample_rate, seed=seed)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    from tqdm import tqdm
    # mean = []
    # std = []
    # train_loader = get_dataloader(split='train', batch_size=10, sample_rate=64000, seed=25)
    # for i, (audio_input, labels) in enumerate(train_loader):
    #     cur_mean = torch.mean(audio_input)
    #     cur_std = torch.std(audio_input)
    #     mean.append(cur_mean)
    #     std.append(cur_std)
    #     print(cur_mean, cur_std)
    # print("-----------------------------")
    # print(np.mean(mean), np.mean(std))

    test_loader = get_dataloader(split='test', batch_size=28, sample_rate=44000, seed=22)
    # print(len(test_loader)*28)
    for item in tqdm(test_loader):
        # print(item['audio_name'])
        pass


