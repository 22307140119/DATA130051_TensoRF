

# 基于 NeRF 的物体重建和新视图合成
可以搜索 `cx` 快速定位到修改的代码。<br>
文件最后包含 精简版总流程（认为已经安装好所有依赖）和 kaggle 平台操作流程（含依赖安装）。


### 环境配置
- 操作系统: `Windows 11` 
- GPU: `NVIDIA GeForce RTX 4060` 
- CUDA: `Cuda V12.1.66` 

按照 `README_TensoRF.md` 操作，注意安装 GPU 版的 `torch` 。

创建并激活环境:
```shell
conda create -n TensoRF python=3.8
conda activate TensoRF
```

安装依赖: 
```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121    # 最后的 cu121 换成自己的 cuda 版本
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips plyfile tensorboard==2.12.0      # tensorboard 版本不能过高
pip install six
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple  # 用于渲染时可视化相机路径
```



### 数据预处理
若已经获取了 .json 文件则跳过该步。
- 运行 COLMAP 得到 `cameras.txt` 等文本文件
- 在 `TensoRF` 下新建 `data` 目录，将文本文件放在 `./data/dataset_name/sparse/` 下，其中 `dataset_name` 换成你的数据集名称。

```shell
# 以下以 deerqq 数据集为例，将 deerqq 换成实际的数据集名称
python ./dataLoader/colmap2nerf.py --images ./data/deerqq/frames --text ./data/deerqq/sparse --out ./data/deerqq/frames.json
```

命令行参数说明: 
- `images`: 存放图像的文件夹。
- `text`: 存放 COLMAP 输出的 .txt 文件的文件夹。
- `out`: .json 文件输出到的地址。



### 划分训练集和测试集
若已经获取了 `transforms_test.json`, `transforms_train.json`, `test/`, `train/` 则跳过该步。<br>
得到所有数据的 NeRF 格式的 .json 文件以后，再划分训练集和测试集。 

- 新增文件 `./dataloader/split_train_test.py` ，用于从指定的 .json文件 和 图片文件夹 划分训练集和测试集，划分结果分别存入 `transforms_test.json`, `transforms_train.json`, `test/`, `train/` 。

```shell
python ./dataloader/split_train_test.py --json_path ./data/my_test_data1/frames.json --image_dir ./data/my_test_data1/frames/ --output_dir path/to/outputdir --test_ratio 0.1 --seed 42
```

命令行参数说明: 
- `out`: NeRF 格式的 .json 文件地址。 
- `image_dir`: .json 文件对应的图像目录。 
- `output_dir`: 输出到的目录，可以省略，默认和 `json_path` 指示的文件夹相同，即默认将输出文件和 .json 并列存储。 
- `test_ratio`: 测试集比例，可以省略，默认为 0.2 。 
- `seed`: 随机种子，可以省略，默认为 42 。 



### 训练
```shell
# 若运行 lego 等官方数据集则直接用 lego.txt 等
# 若运行自定义数据集则先检查 your_own_data.txt 配置，可查看 opt.py 获取参数的详细信息
python train.py --config configs/your_own_data.txt
```



### 训练的输出文件
由 `./configs/` 下配置文件中的 `expname` 和 `basedir` 确定输出路径。

##### 模型文件
后缀名为 `.th` 的文件。

##### 图像文件
- `imgs_test_all`: 存放​​测试集​​的渲染结果
- `imgs_vis`: 存放​​训练过程中周期性保存的验证图像​​ 

##### TensorBoard 日志文件​​
文件名中包含 `events.out.tfevents.` 的文件，
内含：训练/验证损失、PSNR、SSIM、LPIPS 等指标。
```shell
# log/tensorf_deerqq_VM 换成实际路径
tensorboard --logdir=log/tensorf_deerqq_VM
```
注意上面的命令在弹出链接后要稍等一会再点，否则可能会缺失数据。



### 渲染 
- 修改了 `train.py` 和 `opt.py` 。 <br>
`train.py` - `render_test()` - `if args.render_path`: <br>
如果对应的 `./dataLoader/xxx.py` 文件中定义了 `render_path` 就按该定义渲染，否则用添加的代码中定义的环形路径。 <br>
命令行参数需要包含 `--render_only 1` 且 `--render_path 1` 。

- 添加了工具文件 `my_tools.py` ，定义可视化相机路径 `c2ws` 的函数，要用到 `matplotlib` 。<br>
可视化结果会自动输出到 `./c2ws_visualization.png` 。

运行渲染 ( 有些 txt 配置文件指定了 `render_test=1`，要手动指定为 0 ，否则会额外运行不必要的测试 ):
```shell
python train.py --config ./configs/deerqq.txt --ckpt ./log/tensorf_deerqq_VM/tensorf_deerqq_VM.th --render_only 1 --render_path 1 --render_test 0 --path_radius 3.0 --path_frames 30 --path_height 0.0
```
OOM 则修改 `renderer.py` line 115 左右的 `rgb_map, _, depth_map, _, _ = renderer()` 调用，减小 `chunk` 大小（原为 8192）。

命令行参数说明: 
- `path_radius`: 半径，默认 3.0 。 
- `path_frames`: 帧数，默认 30 （即输出 1 秒视频）。 
- `path_height`: 相机高度，默认 0.0 。 



### 总流程 (以 deerqq 数据集为例)
得到 COLMAP 输出的文本文件后，和照片一起放到 ./data/dataset_name/ 目录下。
```shell
# 确保在 TensoRF 目录下
conda activate TensoRF
```

##### 转换数据集到 NeRF 格式
```shell
python ./dataLoader/colmap2nerf.py --images ./data/deerqq/frames --text ./data/deerqq/sparse --out ./data/deerqq/frames.json
```

##### 划分训练集和测试集
```shell
python ./dataloader/split_train_test.py --json_path ./data/deerqq/frames.json --image_dir ./data/deerqq/frames/ --test_ratio 0.1
```

##### 训练
```shell
# 检查 configs/your_own_data.txt 的配置
python train.py --config configs/your_own_data.txt
```

##### 查看日志
```shell
tensorboard --logdir=log/tensorf_deerqq_VM
```

##### 渲染
```shell
python train.py --config ./configs/your_own_data.txt --ckpt ./log/tensorf_deerq_VM/tensorf_deerq_VM.th --render_only 1 --render_path 1 --render_test 0 --path_frames 30 --path_radius 3.0 --path_height 1.0
```
OOM 则修改 `renderer.py` line 115 左右的 `rgb_map, _, depth_map, _, _ = renderer()` 调用，减小 `chunk` 大小（原为 8192）。



### kaggle 平台操作流程
- 创建代码项目 `kaggle1` 。
- 代码上传到“数据集”，命名 `TensoRF_kaggle1` ，导入到项目。
- 数据集上传到“数据集”，命名 `TensoRF_deerqq` ，导入到项目。

运行以下命令：
```shell
# 训练
cp -r /kaggle/input/tensorf-kaggle1/ /kaggle/working/
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips plyfile tensorboard==2.12.0
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips plyfile tensorboard==2.12.0
cd /kaggle/working/tensorf-kaggle1
!python train.py --config /kaggle/working/tensorf-kaggle1/configs/your_own_data.txt --datadir /kaggle/input/tensorf-deerqq/ --expname tensorf_deerqq_VM --basedir /kaggle/working/log/

# 若需要一并渲染，则再加上: 
cd /kaggle/working/
mkdir tensorf_deerqq_VM
pip install torch==2.0.0 torchvision==0.15.1 -f https://download.pytorch.org/whl/torch_stable.html
!python /kaggle/working/tensorf-kaggle1/train.py --config /kaggle/working/tensorf-kaggle1/configs/your_own_data.txt --datadir /kaggle/input/tensorf-deerqq/ --ckpt /kaggle/working/tensorf_deerqq_VM/tensorf_deerqq_VM.th --render_only 1 --render_path 1 --render_test 0 --path_frames 150     # 渲染 150 帧
```

