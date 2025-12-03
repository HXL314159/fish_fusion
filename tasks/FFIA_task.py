import torch
import torch.nn as nn
import matplotlib
import csv
import matplotlib.pyplot as plt
# 使用非交互式后端，允许在无显示环境（服务器/远程）直接保存图片到文件
matplotlib.use('Agg')
import os
import logging as log_config
import numpy as np
# 根据名字获取损失函数（此处是 clip_ce ）
from tasks.losses import get_loss_func
# 统一评估器，用来在训练/验证/测试集上跑模型并计算
from tasks.evaluate import Evaluator
import argparse
from tasks.early_stopping import save_model
# 训练时显示进度条
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


# 提供统一的训练、验证与测试流程，负责循环训练、计算指标、保存最优模型、输出混淆矩阵和训练过程的统计到 CSV
# ckpt_dir: 保存模型与图像的目录
def trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir):
    logger = log_config.getLogger()
    # 提示开始训练
    logger.info("Starting new training run")
    # 建立评估器
    evaluator = Evaluator(model=model)
    best_acc = 0
    best_epoch = 0
    loss_func = get_loss_func('clip_ce')
    # 定义 CSV 列头 fieldnames
    # Define the fieldnames for the CSV file
    fieldnames = ['Epoch', 'Train_Loss', 'Train_Accuracy', 'Train_Precision', 'Train_Recall', 'Train_F1',
                  'Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1']
    # Create a list to store the epoch results
    epoch_results = []
    for epoch in range(max_epoch):
        # 累计训练损失
        mean_loss = 0
        # 遍历训练集的每个批次并显示进度条
        for data_dict in tqdm(train_loader):
            # 把批次数据移到设备
            data_dict['mel_data'] = data_dict['mel_data'].to(device)
            data_dict['si_data'] = data_dict['si_data'].to(device)
            # 标签移到设备并转为 long ，用于交叉熵
            data_dict['target'] = data_dict['target'].to(device).long()
            data_dict['rgb_data'] = data_dict['rgb_data'].to(device)
            # 启用训练模式（影响 Dropout/BN 等）
            model.train()
            # 前向计算。你在 models/fusion_model.py 里实现融合逻辑。
            # 目前我之前的版本只用到了 si_data ，但接口适配三模态，未来可以扩展
            output_dict = model(data_dict['mel_data'], data_dict['si_data'], data_dict['rgb_data'])
            target_dict = {'target': torch.argmax(data_dict['target'], dim=1)}
            # 计算损失
            loss = loss_func(output_dict, target_dict)
            # 清空梯度
            optimizer.zero_grad()
            loss.backward()
            # 优化器更新
            optimizer.step()
            loss = loss.item()
            # 累计到 mean_loss
            mean_loss += loss
        # 该 epoch 的平均训练损失
        epoch_loss = mean_loss / len(train_loader)
        # 使用评估器在训练集上评估一次
        train_statistics = evaluator.evaluate_av(train_loader)
        # np.mean: 求均值
        train_acc = np.mean(train_statistics['accuracy'])
        train_precision = np.mean(train_statistics['precision'])
        train_recall = np.mean(train_statistics['recall'])
        train_f1 = np.mean(train_statistics['f1'])

        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")
        if epoch % 1 == 0:
            # 切换到评估模式
            model.eval()
            # 在验证集上评估
            val_statistics = evaluator.evaluate_av(val_loader)
            # 取混淆矩阵（用于画图）
            val_cm = val_statistics['confu_matrix']
            val_acc = np.mean(val_statistics['accuracy'])
            val_precision = np.mean(val_statistics['precision'])
            val_recall = np.mean(val_statistics['recall'])
            val_f1 = np.mean(val_statistics['f1'])

            # 构建本轮的 epoch_result 字典并加入 epoch_results ，用于最终写 CSV
            # Append the epoch results to the epoch_results list
            epoch_result = {
                'Epoch': epoch+1,
                'Train_Loss': epoch_loss,
                'Train_Accuracy': train_acc,
                'Train_Precision': train_precision,
                'Train_Recall': train_recall,
                'Train_F1': train_f1,
                'Val_Accuracy': val_acc,
                'Val_Precision': val_precision,
                'Val_Recall': val_recall,
                'Val_F1': val_f1
            }
            epoch_results.append(epoch_result)
            message = val_statistics['message']
            if val_acc > best_acc:
                # 更新最佳
                best_epoch = epoch
                best_acc = val_acc
                best_cm = val_cm
                # 保存最佳验证混淆矩阵
                ConfusionMatrixDisplay(best_cm).plot()
                plt.title("confusion_matrix")
                # ckpt_dir: Fish_workspace/audio-video-fusion/save_models
                fig_name = ckpt_dir + str(best_epoch) + '.png'
                plt.savefig(fig_name)
                # 保存当前最优模型权重与优化器状态等信息
                save_model(os.path.join(ckpt_dir, 'best.pt'), model, optimizer, best_epoch)
            # 恢复训练模式，继续下一轮
            model.train()
        
        logger.info(f'val_best_acc: {best_acc}, best_epoch: {best_epoch}')

    # Specify the path and filename for the CSV file
    csv_file = 'results.csv'
    
    # Write the epoch results to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header row->写表头
        writer.writeheader()
        
        # Write the epoch results->逐行写每个 epoch 的统计结果
        writer.writerows(epoch_results)
    
    logger.info(f"Results saved to {csv_file}")

    # 测试集测试
    logger.info('Evaluate on the Test dataset_fish')
    model_path = os.path.join(ckpt_dir, 'best.pt')
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # 评估模式
    model.eval()
    test_statistics = evaluator.evaluate_av(test_loader)
    ave_acc = np.mean(test_statistics['accuracy'])
    logger.info(f' accuracy: {ave_acc}')