# Code for read original video to pickle
import decord
import glob
import os
import pickle
# import dtk.transforms as dtf
from tqdm import tqdm
import cv2
import numpy as np
from torchvision import transforms
import torch


def preprocess_video(video_path):
    # Open the video file
    vr = decord.VideoReader(video_path, height=256, width=256)

    # 读取所有的帧并将他们转换为数组
    frames = [frame.asnumpy() for frame in vr]

    # Resize frames to 256x256 pixels using OpenCV
    resized_frames = [cv2.resize(frame, (256, 256)) for frame in frames]

    # Convert frames to tensors
    tensor_frames = [transforms.ToTensor()(frame) for frame in resized_frames]

    # Create a transformation pipeline for data augmentation
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ])

    # Apply data augmentation to the frames
    augmented_frames = [transform(frame) for frame in tensor_frames]

    # Return the list of augmented frames
    return augmented_frames


def my_preprocess_video(video_path, num_frames = 8):
    # 打开文件
    vr = decord.VideoReader(video_path, height=256, width=256)

    # 获取视频总帧数
    total_frames = len(vr)

    # 计算要抽取到的帧索引
    if total_frames < num_frames:
        # 如果视频帧数不够，抽取所有帧
        frame_indices = list(np.arange(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    # 只读取指定帧
    selected_frames = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        selected_frames.append(frame)

    # 使用OpenCV将帧大小调整为256x256像素
    resized_frames = [cv2.resize(frame, (256, 256)) for frame in selected_frames]

    # 将帧转化为张量
    tensor_frames = [transforms.ToTensor()(frame) for frame in resized_frames]

    # 创建用于数据增强的转换流程
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip()
    ])

    # 对每一帧使用数据增强
    augmented_frames = [transform(frame) for frame in tensor_frames]

    # 确保返回的是八帧
    while len(augmented_frames) < num_frames:
        # 用最后一帧填充，如果没有帧则创建空白帧
        if augmented_frames:
            augmented_frames.append(augmented_frames[-1])
        else:
            blank_frame = torch.zeros(3, 224, 224)
            augmented_frames.append(blank_frame)

    return augmented_frames


# 在windows处理视频
def preprocess_video_windows(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 注意 OpenCV 是 BGR
        frames.append(frame)

    cap.release()

    tensor_frames = [transforms.ToTensor()(f) for f in frames]

    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ])

    augmented_frames = [transform(f) for f in tensor_frames]
    return augmented_frames


def run_preprocess_video(root_dir, output_dir, num_frames):
    # Get a list of all video files in the directory hierarchy
    video_files = glob.glob(os.path.join(root_dir, '**/*.mp4'), recursive=True)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a tqdm progress bar
    progress_bar = tqdm(video_files, desc='Processing videos', unit='video')

    # Iterate over each video file
    for video_file in progress_bar:
        # Get the original path and filename without the extension
        path_without_extension = os.path.splitext(video_file)[0]

        # Create the output directory structure using the original file path
        relative_path = os.path.relpath(path_without_extension, root_dir)
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        # Preprocess the video and get the augmented frames
        augmented_frames = my_preprocess_video(video_file, num_frames)

        # Create the pickle file path using the original file path
        pickle_file = os.path.join(output_subdir, os.path.basename(path_without_extension) + '.pickle')

        # Save the augmented frames as a pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(augmented_frames, f)

        # Update the progress bar
        progress_bar.set_postfix({'Processed': os.path.basename(video_file)})

    # Close the progress bar
    progress_bar.close()


root_dir = '/root/shared-nvme/DATASETS/fish/video_dataset'
output_dir = '/root/shared-nvme/DATASETS/fish/video_pickle'
num_frames = 8
run_preprocess_video(root_dir, output_dir, num_frames)
