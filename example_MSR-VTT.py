import torch
import os
from frechet_video_distance import frechet_video_distance
from diffusers.utils import load_image
import torchvision.io as io
import imageio
from torchvision.transforms import ToTensor
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils.util import natural_sort_key, extract_frames, frames_to_tensors


def get_MSRVTT_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--number_of_videos', type=int, default=101,
                        help='The number of videos.')
    parser.add_argument('--video_length', type=int, default=16,
                        help='The number of videos.')
    parser.add_argument('--models_weights', type=str, default="./pytorch_i3d_model/models/rgb_imagenet.pt",
                        help='The path of the models weights.')
    parser.add_argument('--msrvtt_path', type=str, default="/home/lzh/code/todo/datasets/MSR_VTT",
                        help='The path of the msrvtt_path dataset.')
    parser.add_argument('--result_path', type=str, default="/home/lzh/code/todo/generative-models-main/quantity-speed",
                        help='The path of the result.')
    parser.add_argument('--msrvtt_path_width', type=int, default=320,
                        help='The width of the msrvtt_path dataset.')
    parser.add_argument('--msrvtt_path_height', type=int, default=240,
                        help='The height of the msrvtt_path dataset.')
    parser.add_argument('--result_width', type=int, default=512,
                        help='The width of the result.')
    parser.add_argument('--result_height', type=int, default=512,
                        help='The height of the result.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use. Like cuda, cuda:0 or cpu')

    # 解析命令行参数
    args = parser.parse_args()
    return args


def main(args):
    # 设置相关参数
    NUMBER_OF_VIDEOS = args.number_of_videos
    VIDEO_LENGTH = args.video_length
    PATH_TO_MODEL_WEIGHTS = args.models_weights
    MSRVTT_WIDTH = args.msrvtt_width
    MSRVTT_HEIGHT = args.msrvtt_height
    RESULT_WIDTH = args.result_width
    RESULT_HEIGHT = args.result_height
    # 设置文件夹的路径
    MSRVTT_PATH = args.msrvtt_path
    RESULT_PATH = args.result_path
    # 获取设备
    DEVICE = args.device
    # 初始化两个张量
    first_set_of_videos = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, MSRVTT_HEIGHT, MSRVTT_WIDTH, 3, requires_grad=False).to(DEVICE)
    second_set_of_videos = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, RESULT_HEIGHT, RESULT_WIDTH, 3, requires_grad=False).to(DEVICE) * 255

    # ----------------------------------------赋值第一个张量----------------------------------------
    # 找出测试的video_list
    video_list = []
    msr_vtt_path = '/home/lzh/code/todo/Text2Video-Zero/video_inference_times-500.csv'
    with open(msr_vtt_path, 'r', encoding='utf-8') as csvfile:
        # 创建一个CSV阅读器
        csv_reader = csv.reader(csvfile)
        # 跳过表头（第一行）
        next(csv_reader)
        # 遍历CSV文件的每一行
        for row in csv_reader:
            video_list.append(row[0])
            # breakpoint()
    cnt = 0

    # 遍历MSRVTT数据集的文件夹及其子文件夹中的所有文件
    for index1, sub_dir in enumerate(tqdm(sorted(os.listdir(MSRVTT_PATH)))):
        video_path = os.path.join(MSRVTT_PATH, sub_dir)
        # breakpoint()
        # 只有在测试表格中的视频才进行验证
        if sub_dir.split('.')[0] in video_list:
            # breakpoint()
            frames = extract_frames(video_path, VIDEO_LENGTH)
            # 将抽取的帧转换为0到255范围内的张量
            frame_tensors = frames_to_tensors(frames)
            frame_tensors = torch.stack(frame_tensors).permute(0, 2, 3, 1).contiguous()
            # breakpoint()
            first_set_of_videos[cnt] = frame_tensors
            cnt += 1
    # --------------------------------------------------------------------------------------------

    # breakpoint()

    # ----------------------------------------赋值第二个张量----------------------------------------
    # 遍历模型生成结果文件夹及其子文件夹中的所有文件
    for index1, sub_dir in enumerate(tqdm(sorted(os.listdir(RESULT_PATH)))):
        sub_path = os.path.join(RESULT_PATH, sub_dir)
        for index2, sub_img in enumerate(sorted(os.listdir(sub_path), key=natural_sort_key)):
            img_path = os.path.join(sub_path, sub_img)
            # breakpoint()
            tensor_image = io.read_image(img_path).permute(1, 2, 0).contiguous()
            second_set_of_videos[index1, index2] = tensor_image
    # --------------------------------------------------------------------------------------------

    # 计算FVD
    fvd = frechet_video_distance(first_set_of_videos, second_set_of_videos, PATH_TO_MODEL_WEIGHTS)
    print("FVD:", fvd)


if __name__ == "__main__":
    args = get_MSRVTT_parser()
    main(args)
