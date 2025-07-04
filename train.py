
import os
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

from my_tools import visualize_camera_path  # cx 20250611


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}.')

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):  # 取出下一个 batch 的下标
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):  # 渲染
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    print(f'Tensorf model loaded.\n')

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        print('Starting render_train')
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test, _, _ = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        print('Starting render_test')
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    # cx 20250610: 添加环形渲染路径生成，原代码见后面多行注释
    if args.render_path:
        if hasattr(test_dataset, 'render_path'):    # 如果数据集已有渲染路径则使用它
            c2ws = test_dataset.render_path
        else:                                       # 否则生成自定义环形路径
            print("Generating ring path for Blender dataset...")
            c2ws = []
            
            # 生成环绕路径，在 opt.py 中新增三个参数: path_radius, path_frames, path_height
            radius = args.path_radius or 3.0    # 默认半径 3.0
            frames = args.path_frames or 30     # 默认 30 帧
            height = args.path_height or 0.0    # 默认高度 0.0
            
            for i in range(frames):
                theta = 2 * np.pi * i / frames  # 按照总帧数计算当前帧的"弧度"，一共转一整圈 (2*np.pi 弧度)
                cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), height])   # 计算当前帧的相机位置 (x, y, z)
                
                forward = cam_pos / np.linalg.norm(cam_pos)    # 设置相机方向指向原点，归一化
                up = np.array([0, 0, 1])            # 世界坐标系的上方向
                right = np.cross(up, forward)
                right /= np.linalg.norm(right)      # X 轴 (Z 轴和相机方向的叉乘，归一化)
                real_up = np.cross(forward, right)
                real_up /= np.linalg.norm(real_up)  # Y 轴，真实的上向量 (相机方向和 X 轴的叉乘，归一化)
                
                c2w = np.eye(4)         # 构建 c2w 矩阵
                c2w[:3, 0] = right      # X 轴方向
                c2w[:3, 1] = real_up    # Y 轴方向
                c2w[:3, 2] = forward    # Z 轴方向
                c2w[:3, 3] = cam_pos    # 相机位置

                # 应用 Blender 到 OpenCV 的转换（确保与数据一致）
                # Y轴反转（解决上下颠倒）
                # Z轴反转（解决前后方向）
                blender2opencv = np.array([[1, 0, 0, 0], [0, -1,  0, 0], [0,  0, -1, 0], [0, 0, 0, 1]])
                c2w = c2w @ blender2opencv

                assert c2w.shape == (4, 4), f"Invalid c2w shape: {c2w.shape}"
                c2ws.append(c2w)
                
                # debugging, 检查相机位置与场景中心的距离
                cam_pos = c2w[:3, 3]  # 获取相机位置
                scene_center = (test_dataset.scene_bbox[0] + test_dataset.scene_bbox[1]) / 2
                # distance = np.linalg.norm(cam_pos - scene_center)
                print(f"Camera position: {cam_pos}, Secne center: {scene_center}")
            
            c2ws = np.array(c2ws)
            path_image = visualize_camera_path(c2ws, test_dataset.scene_bbox)
            print(f"Generated ring path with {len(c2ws)} frames, shape: {c2ws.shape}, visualization saved at {path_image}\n")

        # 使用生成的 c2ws 路径渲染，和原本代码一致
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    
        print(f'Videos saved at {logfolder}/{args.expname}/imgs_path_all/depthvideo.mp4 and {logfolder}/{args.expname}/imgs_path_all/video.mp4')

    '''
    if args.render_path:        # 渲染时的相机路径
        c2ws = test_dataset.render_path     # 从 test_dataset 获取给定的相机路径 c2ws
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    '''
                                
def reconstruction(args):

    # init dataset 数据集初始化，只用了 train 和 test ，没有用到 val
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)  # is_stack=False 表示训练集以射线形式存储
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)     # is_stack=True 表示测试集以完整图像形式存储
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file 初始化日志文件
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters 模型参数初始化
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:   # 初始化 tensorf 模型
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:   # 学习率指数衰减，开 args.n_iters 次根
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))   # Adam 优化器


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    # 正则化相关
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    # 正式训练循环开始
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:  # 一次迭代


        ray_idx = trainingSampler.nextids()     # 随机采样一个 batch 的训练射线和对应的 RGB 值
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)   # MSE 损失


        # loss 正则化
        total_loss = loss
        if Ortho_reg_weight > 0:    # 正交正则化
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:       # L1 正则化
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:     # TV 正则化（密度）
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:         # TV 正则化（外观）
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()            # 反向传播与优化（确定损失函数可以看这里）

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))   # 记录指标
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        # 测试，可以通过 vis_every 控制测试频率
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test, extra_metrics, losses_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=True) # cx 20250617, compute_extra_metrics=True
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            summary_writer.add_scalar('test/ssim', extra_metrics[0], global_step=iteration)         # cx 20250617, for testing ssim
            summary_writer.add_scalar('test/lpips_alex', extra_metrics[1], global_step=iteration)   # cx 20250617, for testing l_a
            summary_writer.add_scalar('test/lpips_vgg', extra_metrics[2], global_step=iteration)    # cx 20250617, for testing l_a
            summary_writer.add_scalar('test/mse', np.mean(losses_test), global_step=iteration)      # cx 20250612, for testing loss



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')

    # 训练完成后的渲染
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test, _, _ = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test, _, _ = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:    # 训练完成后，同样通过 render_path 参数触发渲染
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

