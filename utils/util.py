from torchvision.transforms import ToTensor
from moviepy.editor import VideoFileClip
import re
import torch


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
