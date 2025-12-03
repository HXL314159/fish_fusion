from torch import nn
import torch
import torchaudio


# 将整段视频按固定窗口切分成多个片段，每段送入 S3D 提取特征，再把各片段的特征在时间维上堆叠，得到片段级视频特征序列，便于与音频片段或后续序列模型对齐
class S3DWrapper(nn.Module):
    def __init__(self, s3d_model):
        super(S3DWrapper, self).__init__()
        self.s3d_model = s3d_model

    def forward(self, video_input):
        # video_input形状为：torch.Size([8, 8, 3, 224, 224])
        batch_size, num_frames, channels, height, width = video_input.size()

        # Process each clip separately
        num_clips = num_frames // 8
        video_input = video_input.view(batch_size, num_clips, 8, channels, height, width)
        video_features = []
        for clip_idx in range(num_clips):
            clip_input = video_input[:, clip_idx, :, :, :, :]
            clip_output = self.s3d_model(clip_input)
            video_features.append(clip_output)

        # Combine the outputs from each clip
        video_features = torch.stack(video_features, dim=1)
        # video_features形状为：torch.Size([8, 1, 1024])
        # 1024(Feature Dim):因为3D图像(8,3,224,224)经过S3D网络被压缩成了一个1024维的特征向量
        return video_features


# 按视频片段数把整段音频均匀切分为多段，将每段送入音频编码器（如 Cnn6 ）提取固定维度的嵌入，并在片段维上堆叠，得到片段级的音频特征序列
class AudioProcessor(nn.Module):
    def __init__(self, cnn6):
        super(AudioProcessor, self).__init__()
        self.cnn6 = cnn6

    def forward(self, audio_input, num_clips):
        batch_size, audio_length = audio_input.size()
        clip_length = audio_length // num_clips

        # Compute log-mel spectrograms for each audio clip
        audio_features = []
        for clip_idx in range(num_clips):
            start = clip_idx * clip_length
            end = start + clip_length
            clip_audio = audio_input[:, start:end]
            clip_mel_spectrogram = self.cnn6(clip_audio)
            audio_features.append(clip_mel_spectrogram)

        audio_features = torch.stack(audio_features, dim=1)  # (batch_size, num_clips, n_mels, time)

        return audio_features


class Fish_Fusion(nn.Module):
    def __init__(self, audio_encoder, visual_encoder, output_dim=512, num_categories=4):
        super(Fish_Fusion, self).__init__()

        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder

        self.audio_processor = AudioProcessor(audio_encoder)
        self.video_encoder = S3DWrapper(visual_encoder)

        self.output_dim = output_dim
        self.num_categories = num_categories

        # 被定义但未使用
        self.visual_linear = nn.Linear(1024, self.output_dim)

        self.fusion_model = BalanceMLA(self.audio_encoder, self.visual_encoder, self.output_dim, self.num_categories)

    def forward(self, audio_input, video_input):
        # 处理视频
        # video_input形状为：torch.Size([8, 8, 3, 224, 224])
        visual_features = self.video_encoder(video_input) # 形状为：torch.Size([8, 1, 1024])

        # visual_features = self.visual_linear(video_features)
        # import ipdb; ipdb.set_trace()
        
        # 从视频特征张量中，提取出每个视频被分成的剪辑数量
        num_clips = visual_features.size(1)

        # 处理音频
        # audio_input形状为：torch.Size([8, 128000])
        audio_features = self.audio_processor(audio_input, num_clips) # 形状为：torch.Size([8, 128000])

        # 融合特征
        fused_features = self.fusion_model(audio_features, visual_features) # 形状为：torch.Size([8, 4])
        output_dict = {'clipwise_output': fused_features }
        return output_dict


class ResidualProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual_features = self.linear(x)
        return residual_features


# 做双向跨模态注意力对齐与残差融合，让音频片段与视频片段在语义与时间上相互条件化后再融合
class AttentionBRFF(nn.Module):
    def __init__(self, audio_encoder, visual_encoder, residual_dim=512):
        super(AttentionBRFF, self).__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.residual_dim = residual_dim
        self.audio_residual_projection = ResidualProjection(512, self.residual_dim)
        self.visual_residual_projection = ResidualProjection(1024, self.residual_dim)
        self.cross_modal_attention = nn.MultiheadAttention(residual_dim, num_heads=8, batch_first=True)

    def forward(self, audio_input, visual_input):
        # audio_input形状为：torch.Size([8, 1, 512])，visual_input形状为：torch.Size([8, 1, 1024])
        # 维度投影：用 ResidualProjection 把音频 (512) 与视频 (1024) 各自投到共同维度 (512)
        residual_audio = self.audio_residual_projection(audio_input) # 形状为：torch.Size([8, 1, 512])
        residual_visual = self.visual_residual_projection(visual_input) # 形状为：torch.Size([8, 1, 512])

        # 音频作为 query ，视频作为 key/value 计算注意力，得到 ffuse_a1
        ffuse_a1, _ = self.cross_modal_attention(residual_audio, residual_visual, residual_visual) # 形状为：torch.Size([8, 1, 512])
        # 视频作为 query ，音频作为 key/value 计算注意力，得到 ffuse_v1
        ffuse_v1, _ = self.cross_modal_attention(residual_visual, residual_audio, residual_audio) # 形状为：torch.Size([8, 1, 512])

        # 残差融合
        ffuse_a = ffuse_a1 + residual_audio # 形状为：torch.Size([8, 1, 512])
        ffuse_v =ffuse_v1 + residual_visual # 形状为：torch.Size([8, 1, 512])
        return ffuse_a, ffuse_v


class BalanceMLA(nn.Module):
    def __init__(self, audio_encoder, visual_encoder, output_dim=128, num_categories=10):
        super(BalanceMLA, self).__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.attention_brff = AttentionBRFF(audio_encoder, visual_encoder)
        self.adaptive_decision_fusion = AdaptiveDecisionFusion()
        self.category_level_weighting = CategoryLevelWeighting()
        self.prediction_layer = nn.Linear(output_dim, num_categories)

    def forward(self, audio_input, visual_input):
        ffuse_a, ffuse_v = self.attention_brff(audio_input, visual_input) #1,6,512
        avg_ffuse_a = torch.mean(ffuse_a, dim=1)
        avg_ffuse_v = torch.mean(ffuse_v, dim=1)
        ffu_a = self.prediction_layer(avg_ffuse_a)
        ffu_v = self.prediction_layer(avg_ffuse_v)
        weight_vector, avg_ffuse_a, avg_ffuse_v = self.adaptive_decision_fusion(avg_ffuse_a, avg_ffuse_v)

        fused_representation = self.category_level_weighting(weight_vector, ffu_a, ffu_v)
        return fused_representation


class AdaptiveDecisionFusion(nn.Module):
    def __init__(self):
        super(AdaptiveDecisionFusion, self).__init__()
        self.W = nn.Parameter(torch.randn(4, 8))
        self.bias = nn.Parameter(torch.zeros(4))
        self.pro_layer = nn.Linear(512, 4)
    def forward(self, avg_ffuse_a, avg_ffuse_v):
        
        pro_ffuse_a = self.pro_layer(avg_ffuse_a) 
        pro_ffuse_v = self.pro_layer(avg_ffuse_v) 
        concatenated_features = torch.cat([pro_ffuse_a, pro_ffuse_v], dim=1)  # (batch_size, 2 * features)
        
        weighted_features = torch.matmul(self.W, concatenated_features.transpose(1, 0))  # (output_dim, batch_size)
        weighted_features = weighted_features + self.bias.unsqueeze(1)  # (output_dim, batch_size)
        weight_vector = nn.functional.softmax(weighted_features, dim=0)  # (output_dim, batch_size)
        # import ipdb; ipdb.set_trace()
        return weight_vector, avg_ffuse_a, avg_ffuse_v

class CategoryLevelWeighting(nn.Module):
    def __init__(self):
        super(CategoryLevelWeighting, self).__init__()

    def forward(self, weight_vector, avg_ffuse_a, avg_ffuse_v):
        alpha = weight_vector.transpose(0, 1)  # (batch_size, output_dim)
        fused_representation = (avg_ffuse_a * alpha) + (avg_ffuse_v * (1 - alpha))
        # import ipdb; ipdb.set_trace()
        return fused_representation



if __name__ == '__main__':
    video_input = torch.randn(10, 64, 3, 224, 224)
    batch_size, num_frames, channels, height, width = video_input.size()
    print(f'video_input.size: {video_input.size()}')

    num_clips = num_frames // 8
    print(f'num_clips: {num_clips}')

    video_input = video_input.view(batch_size, num_clips, 8, channels, height, width)
    print(f'video_input.size: {video_input.size()}')

    from Video_model import VideoS3D as s3d_model
    video_features = []
    for clip_idx in range(num_clips):
        clip_input = video_input[:, clip_idx, :, :, :, :]  # (batch_size, num_frames, channels, height, width)
        clip_output = s3d_model(clip_input)
        print(f'clip_idx: {clip_idx}, clip_output.shape: {clip_output.shape}')
        video_features.append(clip_output)

