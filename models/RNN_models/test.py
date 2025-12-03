# function：construct sample,切分得到patch,储存每个patch的坐标值
# input: image：torch.size(103, 610, 340)
#         window_size：27
# output：pad_image, batch_image_indices

import torch
import torch.nn as nn
from torchvision import transforms


def construct_sample(image, window_size=27):
    # 先输入照片的通道数等数据的指标
    channel, height, width = image.shape

    half_window = int(window_size // 2)  # 13
    # 使用输入边界的复制值来填充
    pad = nn.ReplicationPad2d(half_window)  # 上下左右伸展13单位值，就是26
    # uses (padding_left, padding_right,padding_top, padding_bottom)

    pad_image = pad(image.unsqueeze(0)).squeeze(0)  # torch.Size([103, 636, 366])

    # 用数组存储切分得到的patch的坐标
    # torch.Size([207400, 4])
    batch_image_indices = torch.zeros((height * width, 4), dtype=torch.long)

    t = 0
    for h in range(height):
        for w in range(width):
            batch_image_indices[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_image, batch_image_indices


if __name__ == '__main__':
    input = torch.randn(103, 610, 340)
    out = construct_sample(input)
    print(out.shape[0])
