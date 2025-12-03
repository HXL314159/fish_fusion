import librosa.display
import torch
import torch.nn as nn
import os
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from models.model_zoo.modules import init_bn

# 把原始时域波形转为时频域的 Log-Mel 频谱特征，并在训练模式下做轻量的谱增强；为后续 Cnn6 等音频编码器提供标准化的输入张量
class Audio_Frontend(nn.Module):
    """
    Wav2Mel transformation & Mel Sampling frontend
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(Audio_Frontend, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor-将时域信号转为幅度谱（STFT → magnitude spectrogram）。常见输出维度为 (batch, 1, time_steps, freq_bins)
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor-对幅度谱做 Mel 变换并取对数（或对数近似），输出 (batch, 1, time_steps, mel_bins)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter-频谱增强（遮挡部分时间片或频带），只在训练时使用
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        # import ipdb; ipdb.set_trace()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins) 100,1,251,1025
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) 100,1 251,64

        # TODO expand 251 to 256
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)

        x = x.transpose(1, 3)  # 100,64,251,1
        # 批归一化
        x = self.bn0(x)
        x = x.transpose(1, 3)  # 100,1,251,64
        # 训练时做谱域增强
        if self.training:
            x = self.spec_augmenter(x)
        # x = x.transpose(2, 3)
        # x = x.squeeze(1)

        # (batch, 1, time_steps_padded, mel_bins)
        return x


# class AudioModel(nn.Module):
#     def __init__(self, frontend, backbone, **kwargs):
#         super().__init__(**kwargs)
#         self.frontend = frontend
#         self.backbone = backbone
#
#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         # """
#
#         clipwise_output = self.backbone(self.frontend(input))
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict


class Audio_Model(nn.Module):
    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend
        # self.backone = load_mobilevit_weights(backbone, file_weight='/vol/research/Fish_tracking_master/FishMM/pretrained_models/xxsmodel_best.pth.tar')['state_dict']
        # self.backbone = backbone
        self.encoder = backbone
        old_pretrained_encoder = torch.load('/vol/research/Fish_tracking_master/FishMM/pretrained_models/xxsmodel_best.pth.tar')['state_dict']
        # new_shape = torch.mean(old_pretrained_encoder['module.stem.0.weight'], dim=1)
        # new_shape = new_shape.unsqueeze(1)
        dict_new = self.encoder.state_dict().copy()
        # old_pretrained_encoder['module.stem.0.weight'] = new_shape
        pretrained_encoder = {k.replace('module.', ''): v for k, v in old_pretrained_encoder.items()}
        trained_list = [i for i in pretrained_encoder.keys() if not ('fc' in i or i.startswith('stem.0'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.encoder.load_state_dict(dict_new)

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        # """

        clipwise_output = self.encoder(self.frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict


class AudioModel_Cnn6(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, backbone, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        self.backbone = backbone
        pretrained_audio_encoder = torch.load('/vol/research/Fish_tracking_master/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        for name, param in self.audio_encoder.named_parameters():

            # if "fc_audioset" not in name:
            param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))
        clipwise_output = self.backbone(output)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict


class AudioModel_MV2(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        pretrained_audio_encoder = torch.load('/vol/research/Fish_tracking_master/FishMM/pretrained_models/MV2/audio_best.pt')['model_state_dict']
        dict_new = self.audio_encoder.state_dict().copy()
        pretrained_encoder = {k.replace('backbone.', ''): v for k, v in pretrained_audio_encoder.items()}
        pretrained_encoder = {k.replace('frontend.', ''): v for k, v in pretrained_encoder.items()}
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))
        return output


class AudioModel_Panns6(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre

        # Resolve pretrained weights path relative to repository root
        # os.path.dirname：取父目录 → 上一层目录
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(repo_root, 'pretrained_models', 'PANNs', 'Cnn6.pth')
        weights_path = os.environ.get('CNN6_PATH', default_path)

        try:
            pretrained_audio_encoder = torch.load(weights_path, map_location='cpu')['model']
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Pretrained Cnn6 weights not found at '{weights_path}'. "
                f"Please place weights at '{default_path}' or set env 'CNN6_PATH'."
            ) from e

        # audio_encoder是backbone_pre，backbone_pre是Cnn6
        # 获取当前 self.audio_encoder 的参数/缓冲区映射，并做一次浅拷贝，得到一个可编辑的权重字典副本
        # 这里dict_new结构是和pretrained_audio_encoder结构是一样的，都是权重字典
        dict_new = self.audio_encoder.state_dict().copy()

        # 从预训练权重的键集合中挑选需要加载到编码器的层，排除分类头和前端特征提取相关的层
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc_audioset' in i or i.startswith('spec') or i.startswith('logmel'))]

        # 从预训练权重中“按键名选择性覆盖”当前模型的权重字典，然后一次性把这份混合字典加载到 self.audio_encoder ，实现只加载编码器相关层、跳过不兼容的层
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        # audio_encoder是Cnn6
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     # 将该参数标记为不参与梯度计算与更新。反向传播不会为它计算梯度，优化器也不会更新它，相当于“冻结”该参数
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """
        # input形状为：torch.Size([8, 128000])
        audio_output = self.audio_encoder(self.audio_frontend(input)) # 形状为：torch.Size([8, 512])
        # output_dict = {'clipwise_output': clipwise_output}
        return audio_output


class AudioModel_CM2(nn.Module):
    def __init__(self, frontend_pre, backbone_pre, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = backbone_pre
        self.audio_frontend = frontend_pre
        pretrained_audio_encoder = torch.load('/vol/research/Fish_tracking_master/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
        dict_new = self.audio_encoder.state_dict().copy()
        # pretrained_audio_encoder = {k:v for k, v in pretrained_audio_encoder.items() if k in dict_new}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        # for name, param in self.audio_encoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """

        output = self.audio_encoder(self.audio_frontend(input))
        return output


class AudioModel(nn.Module):
    def __init__(self, predata, backbone, **kwargs):
        super().__init__(**kwargs)
        self.frontend = predata
        self.backbone = backbone

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        # """

        clipwise_output = self.backbone(self.frontend(input))
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

if __name__ == '__main__':
    input = torch.randn(4, 128000)
    model = Audio_Frontend(sample_rate=64000,
        window_size=2048,
        hop_size=1024,
        mel_bins=128,
        fmin=1,
        fmax=128000)
    print(model(input).shape)

