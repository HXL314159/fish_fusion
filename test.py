import warnings
warnings.filterwarnings("ignore")
import glob
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch


def get_video_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    # path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/mc02229/Fish_av_dataset/video_pk/'
    path = 'F:/DTTASETS/fish/video_dataset/'
    video = []
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path, dir))
        for dir1 in l2:
            video_dir = os.path.join(path, dir, dir1, split, '*.mp4')
            video.extend(glob.glob(video_dir))
    return video

if __name__ == '__main__':
    video_list = get_video_name(split='none')
    print(video_list)
    print(f'数据量：{len(video_list)}')