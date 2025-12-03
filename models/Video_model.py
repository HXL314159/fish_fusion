import torch
import torch.nn as nn
from models.model_zoo.S3D import S3D, load_S3D_weight
import os


class VideoS3D(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)

        # 把传入的 backbone （S3D 模型实例）用指定路径的预训练权重进行加载
        # self.encoder = load_S3D_weight(backbone, file_weight='/vol/research/Fish_tracking_master/fish_fusion/pretrained_models/S3D_kinetics400.pt')

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_weight_dir = os.path.join(base_dir, 'pretrained_models', 'S3D_kinetics400.pt')
        self.encoder = load_S3D_weight(backbone, file_weight=file_weight_dir)

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """
        # input形状为：torch.Size([8, 8, 3, 224, 224])
        video_output = self.encoder(input) # 形状为：torch.Size([8, 8, 3, 224, 224])
        # output_dict = {'clipwise_output': clipwise_output}
        return video_output



class VideoModel(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)

        self.encoder = backbone
        old_pretrained_encoder = torch.load('/vol/research/Fish_tracking_master/FishMM/pretrained_models/video_best.pt')['model_state_dict']
        dict_new = self.encoder.state_dict().copy()
        pretrained_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_encoder.items()}
        trained_list = [i for i in pretrained_encoder.keys() if not ('head' in i or 'pos' in i)]

        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
        self.encoder.load_state_dict(dict_new)

    def forward(self, input):
        """
        Input: (batch_size, data_length)ave_precision
        """
        clipwise_output, embedding = self.encoder(input)
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict
#
#
# class VideoModel(nn.Module):
#     def __init__(self, backbone, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = backbone
#         # old_pretrained_encoder = torch.load('/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/video_best.pt')['model_state_dict']
#         # dict_new = self.encoder.state_dict().copy()
#         # pretrained_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_encoder.items()}
#         # trained_list = [i for i in pretrained_encoder.keys() if not ('head' in i or 'pos' in i)]
#         # for i in range(len(trained_list)):
#         #     dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
#         # self.encoder.load_state_dict(dict_new)
#
#
#     def forward(self, input):
#         """
#         Input: (batch_size, data_length)ave_precision
#         """
#         clipwise_output, _ = self.encoder(input)
#         output_dict = {'clipwise_output': clipwise_output}
#         return output_dict


if __name__ == '__main__':
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(repo_root, 'pretrained_models', 'S3D_kinetics400.pt')
    print(default_path)
