import torch
import torch.nn as nn
import matplotlib
import csv
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import os
import logging as log_config
import numpy as np
from tasks.losses import get_loss_func
from tasks.evaluate import Evaluator
import argparse
from tasks.early_stopping import save_model
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def trainer(model, optimizer, train_loader, val_loader, test_loader, max_epoch, device, ckpt_dir):
    logger = log_config.getLogger()
    logger.info("Starting new training run（开始新一轮epoch的训练）")

    # 创建评估器
    evaluator = Evaluator(model=model)
    best_acc = 0
    best_epoch = 0
    loss_func = get_loss_func('clip_ce')
    # Define the fieldnames for the CSV file
    fieldnames = ['Epoch', 'Train_Loss', 'Train_Accuracy', 'Train_Precision', 'Train_Recall', 'Train_F1',
                  'Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1']
    # 用于存储每一个epoch测试结果
    epoch_results = []
    # 控制训练总轮次；每一轮对训练集完整遍历一次
    for epoch in range(max_epoch):
        mean_loss = 0
        for data_dict in tqdm(train_loader):
            data_dict['video_form'] = data_dict['video_form'].to(device) # torch.Size([8, 128000])
            data_dict['waveform'] = data_dict['waveform'].to(device) # torch.Size([8, 8, 3, 224, 224])
            data_dict['target'] = data_dict['target'].to(device)

            # 切入训练模式
            model.train()
            output_dict1 = model(data_dict['waveform'], data_dict['video_form'])
            target_dict = {'target': data_dict['target']}

            # 计算损失
            loss = loss_func(output_dict1, target_dict)
            loss.backward()

            # 根据梯度更新参数
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            mean_loss += loss
        epoch_loss = mean_loss / len(train_loader)

        # 测试集评估
        train_statistics = evaluator.evaluate_av(train_loader)
        train_acc = np.mean(train_statistics['accuracy'])
        train_precision = np.mean(train_statistics['precision'])
        train_recall = np.mean(train_statistics['recall'])
        train_f1 = np.mean(train_statistics['f1'])
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")

        current_val_acc = 0

        if epoch % 1 == 0:
            # 切到评估模式
            model.eval()
            # 验证集评估
            val_statistics = evaluator.evaluate_av(val_loader)
            val_cm = val_statistics['confu_matrix']
            val_acc = np.mean(val_statistics['accuracy'])

            current_val_acc = val_acc

            # 打印当前验证集正确率和当前epoch
            logger.info(f"current val_acc: {val_acc}, current epoch: {epoch}")

            val_precision = np.mean(val_statistics['precision'])
            val_recall = np.mean(val_statistics['recall'])
            val_f1 = np.mean(val_statistics['f1'])

            # Append the epoch results to the epoch_results list
            epoch_result = {
                'Epoch': epoch + 1, 'Train_Loss': epoch_loss, 'Train_Accuracy': train_acc, 'Train_Precision': train_precision,
                'Train_Recall': train_recall, 'Train_F1': train_f1, 'Val_Accuracy': val_acc, 'Val_Precision': val_precision,
                'Val_Recall': val_recall, 'Val_F1': val_f1
            }
            epoch_results.append(epoch_result)
            message = val_statistics['message']

            # 更新最佳准确度
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_cm = val_cm
                ConfusionMatrixDisplay(best_cm).plot()
                plt.title("confusion_matrix")
                fig_name = ckpt_dir + str(best_epoch) + '.png'
                plt.savefig(fig_name)
                save_model(os.path.join(ckpt_dir, 'best.pt'), model, optimizer, best_epoch)
            # 恢复训练模式
            model.train()

        logger.info(f'val_best_acc: {best_acc}, best_epoch: {best_epoch}')

        # 记录并添加每一个epoch结果
        my_fieldnames = ['current_epoch_num', 'current_train_acc', 'current_valid_acc', 'current_best_epoch', 'current_valid_best_acc']
        csv_file_name = 'results_hxl.csv'
        # 如果文件不存在，就先创建并写入表头
        if not os.path.exists(csv_file_name):
            with open(csv_file_name, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=my_fieldnames)
                writer.writeheader()
        with open(csv_file_name, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=my_fieldnames)
            writer.writerow({'current_epoch_num': epoch, 'current_train_acc': train_acc, 'current_valid_acc': current_val_acc, 'current_best_epoch': best_epoch, 'current_valid_best_acc': best_acc})
        logger.info(f'当前批次：{epoch}，训练集准确率：{train_acc}，验证集准确率：{current_val_acc}，最佳验证集准确率：{best_acc}，最佳验证集批次：{best_epoch}')

    csv_file = 'results.csv'

    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the epoch results
        writer.writerows(epoch_results)

    logger.info("Results saved to %s", csv_file)

    # Test evaluate
    logger.info('Evaluate on the Test dataset_fish（在测试集上评估模型）')
    model_path = os.path.join(ckpt_dir, 'best.pt')
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # 切到评估模式
    model.eval()
    test_statistics = evaluator.evaluate_av(test_loader)
    ave_acc = np.mean(test_statistics['accuracy'])
    logger.info(f' accuracy: {ave_acc}')
