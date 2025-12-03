import warnings
from dataset_fish.dataset import get_dataloader
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
import torch.nn as nn
import os
import time
import logging as log_config
import argparse
from tasks.FFIA_task import trainer
from omegaconf import OmegaConf
from models.fusion_model import fish_fusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--config', type=str, default='config/av/exp2.yaml')
    args = parser.parse_args()
    # 从 args.config 指定的路径读取 YAML 配置文件
    # 将配置文件中的内容解析为一个结构化的 Python 对象（通常是 DictConfig），便于后续访问
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    Training = config['Training']
    batch_size = Training['Batch_size']
    max_epoch = Training['Max_epoch']
    learning_rate = Training['learning_rate']
    seed = Training['seed']
    classes_num = Training['classes_num']
    # 构建并创建模型保存与日志目录
    # workspace: Fish_workspace exp_name: audio-video-fusion
    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    # exist_ok=True: 如果目录已存在，不会报错，继续执行
    os.makedirs(ckpt_dir, exist_ok=True)
    # 设置日志路径
    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 配置 Python 日志系统，设置日志的格式、级别和输出方式
    log_config.basicConfig(
        # 设置日志记录的最低级别，只记录 INFO 及以上级别的日志消息，级别顺序: DEBUG < INFO < WARNING < ERROR < CRITICAL，
        level=log_config.INFO,
        # asctime: 时间戳, levelname: 日志级别, message: 实际的日志内容
        format=' %(asctime)s - %(levelname)s - %(message)s',
        # 日志处理器列表
        handlers=[
            # 文件输出
            log_config.FileHandler(os.path.join(log_dir, '%s-%d.log' % (exp_name, time.time()))),
            # 控制台输出--将日志输出到控制台（标准输出）
            log_config.StreamHandler()
        ]
    )

    logger = log_config.getLogger()

    # 如果有GPU可用使用第3个GPU设备（索引从0开始）
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # 我使用的是单卡
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 实例化分类模型，指定类别数
    Model = fish_fusion(classes_num=classes_num)
    # 将模型迁移到选定设备
    model = Model.to(device)
    # 若多卡可用，使用 DataParallel 包裹并利用所有 GPU
    # model = nn.DataParallel(model, device_ids=[2, 3])
    model = nn.DataParallel(model, device_ids=[0])

    # 创建 Adam 优化器，学习率与动量项来自配置/常用默认
    # 为什么是默认值 (0.9, 0.999)->这是经过大量实验验证的经验值，在大多数深度学习任务中表现良好
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # 训练集
    # num_workers：表示用于“并行加载数据”的子进程数量
    train_loader = get_dataloader(split='train', batch_size=batch_size, seed=seed, num_workers=8)
    # 测试集
    test_loader = get_dataloader(split='test', batch_size=batch_size, seed=seed, num_workers=8)
    # 验证集
    val_loader = get_dataloader(split='val', batch_size=batch_size, seed=seed, num_workers=8)
    logger.info(config)
    logger.info(model)
    # 打印每个数据集的样本数估计
    logger.info(f"Training dataloader: {len(train_loader)* batch_size} samples")
    logger.info(f"Val dataloader: {len(val_loader)* batch_size} samples")
    logger.info(f"Test dataloader: {len(test_loader)* batch_size} samples")
    trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir)