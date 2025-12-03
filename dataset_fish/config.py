# 采样率：表示每秒钟从连续的声音信号中采集多少个离散的数据点（样本）
sample_rate = 22050
# 窗长，每次分析时考虑的样本数量
window_size = 1024
# 定义了当前窗框（Window）与下一个窗框之间向后移动的样本数
hop_size = 512    # So that there are 64 frames per second
mel_bins = 64
# 频率范围
fmin = 1      # Hz
fmax = 14000     # Hz