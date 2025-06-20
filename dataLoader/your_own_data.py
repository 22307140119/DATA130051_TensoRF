import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *


class YourOwnDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis          # 可视化样本数量（-1表示全部）
        self.root_dir = datadir     # 数据集根目录
        self.split = split          # 数据集分割：'train' / 'test' / 'val'
        self.is_stack = is_stack    # 数据是否堆叠存储
        self.downsample = downsample    #  图像下采样率
        self.define_transforms()    # 图像预处理变换，具体定义见后方

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])       # 预设的场景边界框
        # self.scene_bbox = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])       # 预设的场景边界框
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Blender 到 OpenCV 坐标系的变换矩阵
        self.read_meta()            # 是否使用白色背景
        self.define_proj_mat()      # 最近和最远距离

        self.white_bg = True
        # self.near_far = [0.1,100.0]
        self.near_far = [2.0, 6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):    # 核心函数，负责解析数据集的元信息和图像

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)    # 从 transforms_train/test/val.json 中加载元数据


        # 计算相机参数
        w, h = int(self.meta['w']/self.downsample), int(self.meta['h']/self.downsample)
        self.img_wh = [w,h]     # 宽/高 
        self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # 水平焦距 original focal length 
        self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # 垂直焦距 original focal length
        self.cx, self.cy = self.meta['cx'],self.meta['cy']                  # 主点坐标

        # 计算光线方向 ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)   # 归一化
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []


        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#    # 遍历所有图像帧

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv    # 转换到OpenCV坐标系
            c2w = torch.FloatTensor(pose)       # camera 2 world 4x4 相机到世界变换矩阵
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")# 加载并处理图像
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:                    # 下采样
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)                   # 转换为张量 (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)       # 展平 (h*w, 4) RGBA
            if img.shape[-1]==4:    # 混合透明通道
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]  # 保存RGB数据


            # 计算光线
            rays_o, rays_d = get_rays(self.directions, c2w)     # 计算光线原点和方向 both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]   # 拼接 (h*w, 6)


        self.poses = torch.stack(self.poses)
        # 根据 is_stack 参数决定数据存储方式
        if not self.is_stack:       # 展平连接成一个大张量
            self.all_rays = torch.cat(self.all_rays, 0)     # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)     # (len(self.meta['frames])*h*w, 3)

            # self.all_depth = torch.cat(self.all_depth, 0)   # (len(self.meta['frames])*h*w, 3)
        else:                       # 堆叠为多个张量
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])   # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
