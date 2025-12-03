import torch
from torch import nn
from models.model_zoo.panns import PANNS_Cnn10
from models.model_zoo.S3D import load_S3D_weight
from models.model_zoo.S3D import S3D


class ResidualProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, num_sequences, feature_dim = x.size()
        x = x.view(-1, feature_dim)  # Flatten the batch and sequence dimensions
        residual_features = self.linear(x)
        residual_features = residual_features.view(batch_size, num_sequences, -1)  # Restore original shape
        return residual_features


class AttentionBRFF(nn.Module):
    def __init__(self, audio_encoder, visual_encoder, residual_dim=512):
        super(AttentionBRFF, self).__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.audio_residual_projection = ResidualProjection(audio_encoder.output_dim, residual_dim)
        self.visual_residual_projection = ResidualProjection(visual_encoder.output_dim, residual_dim)
        self.cross_modal_attention = nn.MultiheadAttention(residual_dim, num_heads=8, batch_first=True)

    def forward(self, audio_input, visual_input):
        audio_features = self.audio_encoder(audio_input)
        visual_features = self.visual_encoder(visual_input)

        residual_audio = self.audio_residual_projection(audio_features)
        residual_visual = self.visual_residual_projection(visual_features)

        ffuse_a, _ = self.cross_modal_attention(residual_audio, residual_visual, residual_visual)
        ffuse_v, _ = self.cross_modal_attention(residual_visual, residual_audio, residual_audio)

        return ffuse_a, ffuse_v


class BalanceMLA(nn.Module):
    def __init__(self, audio_encoder, visual_encoder, single_modality_model, output_dim=128, num_categories=10):
        super(BalanceMLA, self).__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.single_modality_model = single_modality_model
        self.attention_brff = AttentionBRFF(audio_encoder, visual_encoder)
        self.adaptive_decision_fusion = AdaptiveDecisionFusion(output_dim)
        self.category_level_weighting = CategoryLevelWeighting(num_categories)

    def forward(self, audio_input, visual_input):
        ffuse_a, ffuse_v = self.attention_brff(audio_input, visual_input)
        fused_representation = self.adaptive_decision_fusion(ffuse_a, ffuse_v)
        predictions = self.single_modality_model(fused_representation)
        weighted_predictions = self.category_level_weighting(predictions)
        return weighted_predictions


class AdaptiveDecisionFusion(nn.Module):
    def __init__(self, output_dim):
        super(AdaptiveDecisionFusion, self).__init__()
        self.W = nn.Parameter(torch.randn(output_dim, 1024))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, ffuse_a, ffuse_v):
        avg_ffuse_a = torch.mean(ffuse_a, dim=0)
        avg_ffuse_v = torch.mean(ffuse_v, dim=0)
        concatenated_features = torch.cat([avg_ffuse_a, avg_ffuse_v], dim=1)
        weighted_features = torch.matmul(self.W, concatenated_features.transpose(1, 0)) + self.bias.unsqueeze(0)
        weight_vector = nn.functional.softmax(weighted_features, dim=2)
        fused_representation = torch.bmm(weight_vector[:, 0].unsqueeze(1), ffuse_a.transpose(1, 2)).squeeze(1) + \
                               torch.bmm(weight_vector[:, 1].unsqueeze(1), ffuse_v.transpose(1, 2)).squeeze(1)
        return fused_representation


class CategoryLevelWeighting(nn.Module):
    def __init__(self, num_categories):
        super(CategoryLevelWeighting, self).__init__()
        self.category_weights = nn.Parameter(torch.randn(num_categories))

    def forward(self, predictions):
        weighted_predictions = predictions * self.category_weights.unsqueeze(0).unsqueeze(-1)
        return weighted_predictions


class Fish_Fusion(nn.Module):
    def __init__(self, audio_encoder, video_encoder, output_dim=128, num_categories=4):
        super(Fish_Fusion, self).__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.fusion_model = BalanceMLA(self.audio_encoder, self.video_encoder, None, output_dim, num_categories)

    def forward(self, audio_input, video_input):
        fused_features = self.fusion_model(audio_input, video_input)
        return fused_features


