# cx 20250613
# 指定输入路径：.json 文件路径、图像文件夹路径
# 指定输出到的目录，输出到该目录下的 `transforms_test.json`, `transforms_train.json`, `test/`, `train/` 中
# 指定训练集占总数据集的比例

import argparse
import json
import os
import random
import shutil
from pathlib import Path, PurePath

def parse_args():
    parser = argparse.ArgumentParser(description="division of NeRF dataset")
    parser.add_argument("--json_path", type=str, required=True, help="path to transforms.json")
    parser.add_argument("--image_dir", type=str, required=True, help="dirctory path to images")
    parser.add_argument("--output_dir", type=str, help="output directory")  # 如果没有提供，默认输出到 .json 所在的文件夹中
    parser.add_argument("--test_ratio", type=float, default=0.2, help="proportion of testing set, default=0.2")
    parser.add_argument("--seed", type=int, default=42, help="random seedm random=42")
    args = parser.parse_args()
    return args


# 随机划分训练集和测试集，复制图像到对应目录
def split_dataset(json_path: str, image_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    
    # 读取 .json 文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data["frames"]
    total_frames = len(frames)
    
    # 生成随机下标
    indices = list(range(total_frames))
    random.shuffle(indices)  # 打乱下标顺序
    split_idx = int(total_frames * test_ratio)
    train_indices = indices[split_idx:]
    test_indices = indices[:split_idx]
    # print(train_indices, test_indices)

    # 根据随机下标取出数据
    train_frames = [frames[i] for i in train_indices]
    test_frames = [frames[i] for i in test_indices]
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True) 
    os.makedirs(test_dir, exist_ok=True)
    
    
    # 保存新的 .json 文件
    def save_split(split_name, split_frames):
        # 文件头部的数据
        split_data = data.copy()
        split_data["frames"] = split_frames

        # 更新图像路径
        for frame in split_data["frames"]:
            framename = os.path.basename(frame["file_path"])    # 获取图片本身的名称
            frame["file_path"] = os.path.join(f"{split_name}", framename)
        
        # 写入 .json 文件
        json_path = os.path.join(output_dir, f"transforms_{split_name}.json")
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        return json_path

    train_json_path = save_split("train", train_frames)
    test_json_path = save_split("test", test_frames)
    

    # 复制图像文件到新文件夹
    def copy_images(frame_list, target_dir):
        for frame in frame_list:
            framename = os.path.basename(frame["file_path"])
            src_path = os.path.join(image_dir, framename)
            # src_path += '.png'
            shutil.copy2(src_path, target_dir)
    
    copy_images(train_frames, train_dir)
    copy_images(test_frames, test_dir)


    # 删除 json 文件 "file_path" 字段中的扩展名 (.png)
    def remove_json_extensions(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 移除扩展名
        for frame in data["frames"]:
            old_path = frame["file_path"]
            base_name = os.path.basename(old_path)
            new_name, _ = os.path.splitext(base_name)  # 分离文件名和扩展名
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            frame["file_path"] = new_path
        
        # 覆盖保存修改后的JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    # 处理两个JSON文件
    remove_json_extensions(train_json_path)
    remove_json_extensions(test_json_path)


if __name__ == "__main__":
    args = parse_args()
    if args.test_ratio <= 0 or args.test_ratio >= 1:    # 检查测试集比例是否合法
        raise ValueError("test_ratio must be between 0 and 1")
    if args.output_dir is None:                         # 没有提供输出路径则默认和输入的 .json 文件保存在统一目录下
        args.output_dir = os.path.dirname(args.json_path)
    
    split_dataset(
        args.json_path,
        args.image_dir,
        args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    train_count = len(list(Path(args.output_dir, "train").glob("*.*")))
    test_count = len(list(Path(args.output_dir, "test").glob("*.*")))
    print(f"Output dir: {Path(args.output_dir).resolve()}")
    print(f"Dataset divided with {train_count} pcs for training set, {test_count} pcs for testing set.")
