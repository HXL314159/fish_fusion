import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.modules import ConvBlock, ConvBlock5x5, init_layer
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class PANNS_Cnn6(nn.Module):
    def __init__(self, classes_num):
        
        super(PANNS_Cnn6, self).__init__()
       
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_layer(self.fc_audioset)   
        # self.load_from_ckpt()

    # def load_from_ckpt(self):
    #     pretrained_cnn = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/PANNs/Cnn6.pth')['model']
    #     dict_new = self.state_dict().copy()
    #     trained_list = [i for i in pretrained_cnn.keys() if not ('fc' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
    #     for i in range(len(trained_list)):
    #         dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
    #     self.load_state_dict(dict_new)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        # output_dict = {
        #     'clipwise_output': clipwise_output, 
        #     'embedding': embedding}

        return clipwise_output, embedding


class PANNS_Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        super(PANNS_Cnn10, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        # self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        # init_layer(self.fc_audioset)
        self.load_from_ckpt()  

    def load_from_ckpt(self):
        pretrained_cnn = torch.load('/vol/research/Fish_tracking_master/fish_fusion/pretrained_models/PANNs/Cnn10.pth')['model']
        dict_new = self.state_dict().copy()
        trained_list = [i for i in pretrained_cnn.keys() if not ('fc_audioset' in i or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
        self.load_state_dict(dict_new)

    def forward(self, input):

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)

        output_dict = {'embedding': embedding}
        return output_dict


if __name__ == '__main__':
    model = PANNS_Cnn10()
    print(model(torch.randn(1, 1, 51, 40)))