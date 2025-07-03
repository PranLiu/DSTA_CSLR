
# 在modules/attention.py中实现新的注意力模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicTemporalAttention(nn.Module):
    def __init__(self, channels, max_window_size=9, kernel_sizes=None, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.max_window_size = max_window_size
        self.reduction_ratio = reduction_ratio

        # 默认卷积核大小
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.kernel_sizes = kernel_sizes

        # 窗口大小预测
        self.window_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.window_predictor = nn.Sequential(
            nn.Conv1d(channels, channels//reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels//reduction_ratio, max_window_size, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 多窗口卷积 - 使用指定的卷积核大小
        if len(kernel_sizes) > 0:
            self.window_convs = nn.ModuleList([
                nn.Conv3d(
                    channels, channels,
                    kernel_size=(k, 1, 1),
                    padding=(k//2, 0, 0),
                    groups=channels
                ) for k in kernel_sizes
            ])
        else:
            # 如果没有指定卷积核大小，则使用默认的方式
            self.window_convs = nn.ModuleList([
                nn.Conv3d(
                    channels, channels,
                    kernel_size=(2*i+1, 1, 1),
                    padding=(i, 0, 0),
                    groups=channels
                ) for i in range(1, max_window_size+1)
            ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        # 时间门控
        self.temporal_gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        # 预测每个位置的窗口大小权重
        x_pool = self.window_pool(x)  # [B, C, T, 1, 1]
        x_pool = x_pool.squeeze(-1).squeeze(-1)  # [B, C, T]

        # 如果使用指定的卷积核大小
        if len(self.kernel_sizes) > 0:
            # 直接应用每个卷积核
            multi_scale_feats = []
            for conv in self.window_convs:
                feat = conv(x)
                multi_scale_feats.append(feat)

            # 平均融合多尺度特征
            fused_feat = sum(multi_scale_feats) / len(multi_scale_feats)
        else:
            # 使用动态窗口大小预测
            window_weights = self.window_predictor(x_pool)  # [B, max_window_size, T]

            # 应用不同窗口大小的卷积
            multi_scale_feats = []
            for i, conv in enumerate(self.window_convs):
                feat = conv(x)
                # 应用对应窗口的权重
                weight = window_weights[:, i:i+1, :].unsqueeze(-1).unsqueeze(-1)  # [B, 1, T, 1, 1]
                weighted_feat = feat * weight
                multi_scale_feats.append(weighted_feat)

            # 融合多尺度特征
            fused_feat = sum(multi_scale_feats)

        # 应用时间门控
        gate = self.temporal_gate(fused_feat)
        out = fused_feat * gate

        # 最终融合
        out = self.fusion(out)

        return out
