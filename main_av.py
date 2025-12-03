import warnings
from dataset_fish.AV_dataset import get_dataloader
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
# 将 PyTorch 的张量共享策略从默认的 shared_memory 更改为 file_system
# 这样，进程间将通过在磁盘上创建临时文件来共享数据，而不是使用有限的 /dev/shm 内存
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import os
import time
import logging as log_config
import argparse
from models.Audio_model import Audio_Frontend
from models.Audio_model import AudioModel_Panns6
from models.model_zoo.Cnn6 import Cnn6
from models.model_zoo.S3D import S3D
from models.Video_model import VideoS3D
from tasks.AV_Fusion_task import trainer
from omegaconf import OmegaConf
from models.AV_model import Fish_Fusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/av/exp2.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    workspace = config['Workspace']
    exp_name = config['Exp_name']
    Training = config['Training']
    Model = config['Model']
    audio_parameters = config['Audio_features']
    audio_type = Model['audio_name']
    video_type = Model['video_name']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    sample_rate = audio_parameters['sample_rate']
    classes_num = Training['classes_num']
    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir, '%s-%d.log' % (exp_name, time.time()))),
            log_config.StreamHandler()
        ]
    )

    logger = log_config.getLogger()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    frontend = Audio_Frontend(**audio_parameters)
    # 把配置里的字符串动态转换成类对象，目前audio_name: Cnn6
    Model = eval(audio_type)
    audio_backbone = Model()
    # 音频编码器 -> frontend: Audio_Frontend, audio_backbone: Cnn6
    audio_encoder = AudioModel_Panns6(frontend_pre=frontend, backbone_pre=audio_backbone).to(device)

    # video_type: S3D
    Model_video = eval(video_type)
    video_backbone = Model_video()
    # 视频编码器 -> video_backbone: S3D
    video_encoder = VideoS3D(backbone=video_backbone).to(device)

    # 融合模态模型
    model = Fish_Fusion(audio_encoder=audio_encoder,  visual_encoder=video_encoder).to(device)

    # 多GPU训练
    # model = nn.DataParallel(model, device_ids=[2, 3])

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 加载训练，测试和验证数据集
    train_loader = get_dataloader(split='train', batch_size=batch_size, seed=seed, sample_rate=sample_rate, num_workers=8)
    test_loader = get_dataloader(split='test', batch_size=batch_size, seed=seed, sample_rate=sample_rate, num_workers=8)
    val_loader = get_dataloader(split='val', batch_size=batch_size, seed=seed, sample_rate=sample_rate, num_workers=8)

    logger.info(config)
    logger.info(model)
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")

    # 启动训练
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir)