import torch
from torch import nn
# 导入音频编码器（PANNs 系列的简化版 Cnn6）
from models.model_zoo.Cnn6 import Cnn6
# 导入轻量级图像编码器 MobileNetV2
from models.model_zoo.MobileNetV2 import MobileNetV2
# 导入 ResNet18 作为 RGB 图像编码器
from models.model_zoo.ResNet_18 import ResNet18


# 继承 nn.Module
class fish_fusion(nn.Module):
    def __init__(self, classes_num, **kwargs):
        super().__init__(**kwargs)
        self.cnn6 = Cnn6(classes_num=classes_num)
        self.MobilenetV2 = MobileNetV2(classes_num=classes_num)
        self.resnet18 = ResNet18(num_classes=classes_num)
        # 融合后的降维全连接层（设计为将三个 512 维特征拼在一起得到 1536，再压缩到 512）
        self.fc1 = nn.Linear(1536, 512)
        # 最终分类层，将 512 维向量映射到 4 类的 logits
        # 注意：这里把输出维度硬编码为 4 ，如果配置里的 classes_num 不是 4，会有不一致。更合理是用 classes_num
        # self.classifer = nn.Linear(512, 4)
        self.classifer = nn.Linear(512, classes_num)


    # mel ：音频的 Log-mel 频谱，形状通常是 [B, 130, 64] （数据集中使用的设置），
    # 若要喂给 Cnn6 ，往往需要扩展一个通道维变成 [B, 1, 130, 64]
    # si_data ：SI 图像，经过 transforms 后是 [B, 3, 224, 224] 的 float 张量
    # rgb_data ：RGB 图像，形状与 si_data 相同
    def forward(self, mel, si_data, rgb_data):
        # B = batch_size = 2 (批大小)，H = height = 130 (频谱图的高度/时间帧数)，W = width = 64 (频谱图的宽度/频率bin数)
        # 把 [B, H, W] 变成 [B, 1, H, W] ，增加了通道维度，符合 CNN 输入
        # mel = mel.unsqueeze(1)
        # 得到音频特征 [B, 512]
        # mel_output = self.cnn6(mel)  # 2,512
        # 用 SI 图像编码器，得到 [B, 512]
        si_output = self.MobilenetV2(si_data)  #2,512
        # 得到 RGB 特征 [B, 512]
        # rgb_output = self.resnet18(rgb_data)  #2, 512
        # 把三个特征拼接为 [B, 1536]
        # fusion_embedding = torch.cat((mel_output, si_output, rgb_output), dim=1) #2, 512 x 3
        # 降维到 [B, 512]
        # out = self.fc1(fusion_embedding)

        # only audio
        # 用分类层得到 [B, 4] 的 logits
        clipwise_output = self.classifer(si_output)
        # 返回字典，兼容训练循环里对 clipwise_output 的读取
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict 












