import os
import glob


def get_audio_file(video_files):
    dir, audio_name = os.path.split(video_files)
    audio_id = (audio_name.split('.')[0])[9:]
    dir1, audio_name1 = os.path.split(dir)
    dir2, audio_name2 = os.path.split(dir1)
    dir3, audio_name3 = os.path.split(dir2)
    # audio_files = glob.glob(f'/vol/research/Fish_tracking_master/Fish_av_dataset/audio_dataset/{audio_name3}/{audio_name2}/{audio_name1}/*.wav')
    audio_files = glob.glob(f'/root/shared-nvme/DATASETS/fish/audio_dataset/{audio_name3}/{audio_name2}/{audio_name1}/*.wav')
    for audio_file in audio_files:
        dir4, audio_name = os.path.split(audio_file)
        video_id = (audio_name.split('.')[0])[9:]
        if str(video_id) == str(audio_id):
            return os.path.join(dir4, audio_name)


if __name__ == '__main__':
    video_file = 'F:/桌面\音视频数据/video/medium/21_video_10.pickle'
    dir, audio_name = os.path.split(video_file)
    print(f'dir: {dir}')
    print(f'audio_name: {audio_name}')
    audio_id = (audio_name.split('.')[0])[9:]
    print(f'audio_id: {audio_id}')

    dir1, audio_name1 = os.path.split(dir)
    dir2, audio_name2 = os.path.split(dir1)
    dir3, audio_name3 = os.path.split(dir2)
    print(f'dir1: {dir1}')
    print(f'audio_name1: {audio_name1}')
    print(f'dir2: {dir2}')
    print(f'audio_name2: {audio_name2}')
    print(f'dir3: {dir3}')
    print(f'audio_name3: {audio_name3}')

    audio_files = glob.glob(f'F:/桌面/音视频数据/audio/{audio_name1}/*.wav')
    print(audio_files)

