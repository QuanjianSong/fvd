import torch
import os
from frechet_video_distance import frechet_video_distance
from diffusers.utils import load_image
import torchvision.io as io
import imageio
from torchvision.transforms import ToTensor
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import re


NUMBER_OF_VIDEOS = 101
VIDEO_LENGTH = 16
PATH_TO_MODEL_WEIGHTS = "./pytorch_i3d_model/models/rgb_imagenet.pt"

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def extract_frames(video_path, num_frames):
    clip = VideoFileClip(video_path)
    total_frames = int(clip.fps * clip.duration)

    # 确保num_frames不超过视频的总帧数
    num_frames = min(num_frames, total_frames)

    frame_indices = torch.arange(0, num_frames).long()
    # frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()

    frames = []
    # breakpoint()
    for idx in frame_indices:
        frame = clip.get_frame(idx * 15)
        frames.append(frame)

    return frames

def frames_to_tensors(frames):
    to_tensor = ToTensor()

    tensors = []
    for frame in frames:
        # 转换为张量
        tensor = to_tensor(frame)

        # 将张量值乘以255并转换为整数类型
        tensor = (tensor * 255).to(torch.uint8)

        # 添加到列表中
        tensors.append(tensor)

    return tensors

def frames_to_tensors(frames):
    to_tensor = ToTensor()
    tensors = []
    for frame in frames:
        # 转换为张量
        tensor = to_tensor(frame)
        # 将张量值乘以255并转换为整数类型
        tensor = (tensor * 255).to(torch.uint8)
        # 添加到列表中
        tensors.append(tensor)

    return tensors

def main():
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    first_set_of_videos = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 240, 320, 3, requires_grad=False)
    second_set_of_videos = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, 576, 1024, 3, requires_grad=False) * 255
    
    
    num_frames = VIDEO_LENGTH
    # ----------------------------------------赋值第一个张量----------------------------------------
    # 设置文件夹的路径
    folder_path = '/home/lzh/code/todo/datasets/UCF-101'
    # 遍历文件夹及其子文件夹中的所有文件
    for index1, sub_dir in enumerate(tqdm(sorted(os.listdir(folder_path)))):
        page_dir = os.path.join(folder_path, sub_dir)
        for index2, sub_video in enumerate(sorted(os.listdir(page_dir))):
            video_path = os.path.join(page_dir, sub_video)
            frames = extract_frames(video_path, num_frames)
            # 将抽取的帧转换为0到255范围内的张量
            frame_tensors = frames_to_tensors(frames)
            frame_tensors = torch.stack(frame_tensors).permute(0, 2, 3, 1).contiguous()
            # breakpoint()
            first_set_of_videos[index1] = frame_tensors
        

        # breakpoint()
        # output_list.append(sub_dir)
            break # 只拿第一个视频
    # --------------------------------------------------------------------------------------------

    # breakpoint()

    # ----------------------------------------赋值第二个张量----------------------------------------
    # 设置文件夹的路径
    folder_path = '/home/lzh/code/todo/generative-models-main/quantity-speed'
    # 遍历文件夹及其子文件夹中的所有文件
    for index1, sub_dir in enumerate(tqdm(sorted(os.listdir(folder_path)))):
        sub_path = os.path.join(folder_path, sub_dir)
        for index2, sub_img in enumerate(sorted(os.listdir(sub_path), key=natural_sort_key)):
            img_path = os.path.join(sub_path, sub_img)
            # print(img_path)
            # breakpoint()
            
            tensor_image = io.read_image(img_path).permute(1, 2, 0).contiguous()

            second_set_of_videos[index1, index2] = tensor_image
            
            # breakpoint()
            # output_list.append(sub_dir)
    # --------------------------------------------------------------------------------------------




    fvd = frechet_video_distance(first_set_of_videos, second_set_of_videos, PATH_TO_MODEL_WEIGHTS)
    print("FVD:", fvd)


if __name__ == "__main__":
    main()
