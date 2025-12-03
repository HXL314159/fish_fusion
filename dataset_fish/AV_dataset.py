import warnings

from utils.MultiHeadAttention import batch_size

warnings.filterwarnings("ignore")
import glob
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import pickle
from utils.get_audioname import get_audio_file
import librosa
from scipy.signal import resample


def get_video_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    # path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/mc02229/mc02229/Fish_av_dataset/video_pk/'
    path = '/root/shared-nvme/DATASETS/fish/video_pickle'
    video = []
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path, dir))
        for dir1 in l2:
            video_dir = os.path.join(path, dir, dir1, split, '*.pickle')
            video.extend(glob.glob(video_dir))
    return video


def load_audio(path, sr=None):
    # 使用原始采样率
    y, _ = librosa.load(path, sr=None)
    y = resample(y, num=sr*2)
    return y


def data_generator(seed, test_sample_per_class):
    """
    class to label mapping:
    none: 0
    strong: 1
    middle: 2
    weak: 3
    """
    random_state = np.random.RandomState(seed)
    splits = ['strong', 'medium', 'weak', 'none']
    train_dict, test_dict, val_dict = [], [], []

    for split in splits:
        # 获取对应分类，例如medium分类下的所有video文件名（包括路径），文件名后缀为*.pickle
        video_list = get_video_name(split=split)
        random_state.shuffle(video_list)

        # 划分数据
        test_samples = video_list[:test_sample_per_class]
        val_samples = video_list[test_sample_per_class:2 * test_sample_per_class]
        train_samples = video_list[2 * test_sample_per_class:]

        # 数据打上标签
        # 形状似如：[['F:/DTTASETS/fish/video_dataset\\2022_6_19\\AM_70\\strong\\19_video_51.mp4', 0]]
        for wd in train_samples:
            label = splits.index(split)
            train_dict.append([wd, label])

        for wd in test_samples:
            label = splits.index(split)
            test_dict.append([wd, label])

        for wd in val_samples:
            label = splits.index(split)
            val_dict.append([wd, label])

    random_state.shuffle(train_dict)

    return train_dict, test_dict, val_dict


class Fish_Video_Dataset(Dataset):
    def __init__(self, seed, split, sample_rate=64000):
        """
        split: train or test
        if sample_rate=None, read audio with the default sr
        """
        self.sample_rate = sample_rate
        self.seed = seed
        self.split = split
        train_v_dict, test_v_dict, val_v_dict = data_generator(self.seed, test_sample_per_class=670)
        if self.split == 'train':
            self.data_dict = train_v_dict
        elif self.split == 'test':
            self.data_dict = test_v_dict
        elif self.split == 'val':
            self.data_dict = val_v_dict

    def __len__(self):
        return len(self.data_dict)

    # 真正获取数据的部分
    def __getitem__(self, index):
        video_name, target = self.data_dict[index]
        # 转成one_hot，维度为4
        target = np.eye(4)[target] # 形状为：(4)

        # 从文件中读取视频pickle文件
        with open(video_name, 'rb') as f:
            data = pickle.load(f)
        vr = torch.stack(data)

        # 在一个视频的n个帧中随机抽取八帧
        # full_vid_length = len(vr)
        # X = np.arange(0, full_vid_length)
        # Y = sorted(np.random.choice(X, 8, replace=False))
        # # 多维张量索引简写
        # vf = vr[Y, ...]
        # 不随机取帧
        vf = vr # 形状为：(8, 3, 224, 224)

        # 获取音频原始数据
        wav_name = get_audio_file(video_name)
        wav = load_audio(wav_name, sr=self.sample_rate)
        wav = np.array(wav) # 形状为：[128000]

        # 整合音视频数据
        data_dict2 = {'audio_name': wav_name, 'waveform': wav}
        data_dict1 = {'video_name': video_name, 'video_form': vf, 'target': target}
        data_dict = dict(data_dict1, **data_dict2)

        return data_dict


# 整合batch
def collate_fn(batch):
    # 整合音频
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    wav = torch.FloatTensor(np.array(wav)) # 形状为：torch.Size([8, 128000])

    # 整合视频
    video_name = [data['video_name'] for data in batch]
    vf = torch.stack([data['video_form'] for data in batch]) # 形状为：torch.Size([8, 8, 3, 224, 224])
    # vf = vf.permute(0, 2, 1, 3, 4)

    # 整合真实标签
    target = [data['target'] for data in batch]
    target = torch.FloatTensor(np.array(target)) # 形状为：torch.Size([8, 4])

    # 整合所有音视频数据
    data_dict = {'video_name': video_name, 'video_form': vf, 'target': target, 'audio_name': wav_name, 'waveform': wav}
    return data_dict


def get_dataloader(split, batch_size, sample_rate, seed, shuffle=False, drop_last=False, num_workers=8):
    dataset = Fish_Video_Dataset(split=split, seed=seed, sample_rate=sample_rate)

    # torch.utils.data.DataLoader 会反复调用 dataset.__getitem__() 来取样本
    # batch_size: 每次加载的样本数量
    # drop_last: 如果最后一个 batch 不够大，是否丢弃
    # num_workers: 启用多少个子进程并行加载数据
    # collate_fn: 批量样本的组装函数
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    videos = get_video_name(split='strong')
    print(videos)