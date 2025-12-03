import os
import random
import torch
import numpy as np
# PyTorch 标准数据集和数据加载器
from torch.utils.data import Dataset, DataLoader
# 图像增强与归一化
from torchvision import transforms
# 读 JPG/PNG 图像
from PIL import Image
# 从项目根目录的 config.py 读取音频配置
import config
# 自定义的 Log-mel 特征提取器
from dataset_fish.utils_tools import LogMelExtractor
# 读 .wav 音频，返回波形
import librosa
import time


# 数据拆分与路径拼接
# seed ：设置 Python 的随机种子（仅用于 random.shuffle ），保证每次划分一致
# test_sample_per_class ：每个类别测试样本数（默认 10）
def data_generator(seed=20, test_sample_per_class=10):
    random.seed(seed)
    train_data = []
    test_data = []
    val_data = []

    classes = ['none', 'weak', 'strong', 'medium']
    label_map = {'none': 0, 'weak': 1, 'strong': 2, 'medium': 3}
    # 音频（mel） .wav 所在目录，按类别分子文件夹
    root_dir = 'F:/fish_fusion/dataset/mel'
    # SI 图像根目录
    si_root_dir = 'F:/fish_fusion/dataset/dataset/SI'
    # RGB 图像根目录
    rgb_root_dir = 'F:/fish_fusion/dataset/dataset/RGB'
    # 遍历每个类别
    for class_name in classes:
        # 该类别的音频目录, class_dir: E:/fish_fusion/dataset/mel/ none/weak/strong/medium
        class_dir = os.path.join(root_dir, class_name)
        # os.listdir(class_dir)：列出指定目录class_dir下的所有文件和文件夹名称
        # if filename.endswith('.wav')：筛选出所有以 .wav 结尾的文件
        # os.path.join(class_dir, filename)：将目录路径和文件名拼接成完整路径
        class_paths = [os.path.join(class_dir, filename) for filename in os.listdir(class_dir) if filename.endswith('.wav')]
        # 打乱列表
        random.shuffle(class_paths)

        test_split = test_sample_per_class
        val_split = 2 * test_sample_per_class
        
        # 划分测试集，验证集和训练集（注意这里拿来的只是路径）
        test_samples = class_paths[:test_split]
        val_samples = class_paths[test_split: val_split]
        train_samples = class_paths[val_split:]

        label = label_map[class_name]

        for path in train_samples:
            # os.path.basename(path)：从完整路径中提取文件名
            # os.path.splitext(...)[0]：分割文件名和扩展名，返回(文件名, 扩展名)元组，取第一个元素（文件名部分）
            # .split('_')[1]：用下划线分割文件名，取第二个部分
            # 'RGB_{}.jpg'.format('123')：字符串格式化
            # 'E:/fish_fusion/dataset/dataset/RGB/strong/RGB_123.jpg'
            rgb_path = os.path.join(rgb_root_dir, class_name, 'RGB_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            si_path = os.path.join(si_root_dir, class_name, 'SI_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            train_data.append([path, si_path, rgb_path, label])

        for path in test_samples:
            rgb_path = os.path.join(rgb_root_dir, class_name, 'RGB_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            si_path = os.path.join(si_root_dir, class_name, 'SI_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            test_data.append([path, si_path, rgb_path, label])

        for path in val_samples:
            rgb_path = os.path.join(rgb_root_dir, class_name, 'RGB_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            si_path = os.path.join(si_root_dir, class_name, 'SI_{}.jpg'.format(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
            val_data.append([path, si_path, rgb_path, label])
    random.shuffle(train_data)
    # 返回的每一个包含音频路径，SI 路径，RGB 路径和标签
    return train_data, test_data, val_data


class fish_Dataset(Dataset):
    # seed ：用于 data_generator 的打乱
    # split ：选择 'train'|'test'|'val'
    # test_sample_per_class=200 ：每类测试数默认设置为 200
    def __init__(self, seed, split, test_sample_per_class=200):
        self.sample_rate = config.sample_rate
        self.window_size = config.window_size
        self.hop_size = config.hop_size
        self.mel_bins = config.mel_bins
        self.fmin = config.fmin
        self.fmax = config.fmax
        # Feature extractor
        # 初始化 Log-mel 特征提取器
        # 参数构造 log-mel 提取器。它会把一段波形转为形如 [帧数, mel_bins] 的特征
        self.feature_extractor = LogMelExtractor(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax)
        # 图像增强
        self.transform = transforms.Compose([
            # 随机裁剪并缩放，裁剪区域占原图面积的80%到100%,目标尺寸 256×256
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            # 随机旋转±15度
            transforms.RandomRotation(degrees=15),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 中心裁剪为 224×224
            transforms.CenterCrop(size=224),
            # 转化成张量,#归一化[0,1]（是将数据除以255），
            # transforms.ToTensor（）会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB）
            transforms.ToTensor(),
            #  用 ImageNet 均值和方差做标准化。
            # 均值 [0.485, 0.456, 0.406]：ImageNet 数据集中所有图片在R、G、B三个通道的像素平均值
            # 标准差 [0.229, 0.224, 0.225]：ImageNet 数据集中所有图片在R、G、B三个通道的像素标准差
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        self.split = split
        # train_dict 的整体结构: 
        # train_dict = [
        #     [audio_path_1, si_path_1, rgb_path_1, label_1],
        #     [audio_path_2, si_path_2, rgb_path_2, label_2],
        #     [audio_path_3, si_path_3, rgb_path_3, label_3],
        #     ... 更多样本
        # ]
        # [
        #   path,      # 字符串：音频文件路径，如 'E:/.../mel/strong/sample_123.wav'
        #   si_path,   # 字符串：SI图像文件路径，如 'E:/.../SI/strong/SI_123.jpg'
        #   rgb_path,  # 字符串：RGB图像文件路径，如 'E:/.../RGB/strong/RGB_123.jpg'
        #   label      # 整数：类别标签，如 0, 1, 2, 3
        # ]
        train_dict, test_dict, val_dict = data_generator(seed, test_sample_per_class)
        if self.split == 'train':
            self.data_dict = train_dict
        elif self.split == 'test':
            self.data_dict = test_dict
        elif self.split == 'val':
            self.data_dict = val_dict

    def __len__(self):
        return len(self.data_dict)

    # 真正读取文件内容（发生磁盘 IO）
    # 凡是看到代码里“for batch in train_loader/val_loader/test_loader”这种迭代，
    # 底层就是在调用 fish_Dataset.__getitem__ 来从文件中把音频和图像读到内存里
    def __getitem__(self, index):
        mel_path, si_path, rgb_path, target = self.data_dict[index]
        # 使用 librosa 库加载音频文件,mel_path：音频文件路径, sr=self.sample_rate：指定采样率为 22050 Hz（统一采样率）
        audio, sample_rate = librosa.load(mel_path, sr=self.sample_rate)
        # 将原始音频波形转换为 Log-Mel 频谱特征
        feature = self.feature_extractor.transform(audio)
        # 确保所有音频特征具有相同的维度
        if feature.shape[0] != 130 or feature.shape[1] != 64:
            print(mel_path)
        si_image = Image.open(si_path)
        rgb_image = Image.open(rgb_path)
        # 图像数据增强和标准化
        si_data = self.transform(si_image)
        rgb_data = self.transform(rgb_image)
        # 将整数标签转换为 one-hot 向量
        # np.eye(4) 创建 4×4 的单位矩阵
        target = np.eye(4)[target]
        data_dict = {'mel_data': feature, 'si_data': si_data, 'rgb_data': rgb_data, 'target': target}

        return data_dict


# 自定义打包函数
def collate_fn(batch):
    # 收集音频特征列表
    feature = [data['mel_data'] for data in batch]
    # 把每个样本的 SI 张量堆成 [B, 3, 224, 224]
    si_data = torch.stack([data['si_data'] for data in batch])
    rgb_data = torch.stack([data['rgb_data'] for data in batch])
    # 标签列表，形如 [B, 4]
    target = [data['target'] for data in batch]
    # 把音频特征转为 FloatTensor ，形状约 [B, 130, 64]
    feature = torch.FloatTensor(feature)
    # 标签也转成 FloatTensor
    target = torch.FloatTensor(target)
    return {'mel_data': feature, 'si_data': si_data, 'rgb_data': rgb_data, 'target': target}


def get_dataloader(split,
                   batch_size,
                   seed,
                   shuffle=False,
                   drop_last=False,
                   num_workers=8):

    dataset = fish_Dataset(split=split, seed=seed)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    from tqdm import tqdm
    # train_loader = get_dataloader(split='train', batch_size=10, seed=10, num_workers=8)
    train_loader = fish_Dataset(split='train', seed=20)
    for item in tqdm(train_loader):
        # print(item['mel_path'])
        pass
