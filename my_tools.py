import matplotlib.pyplot as plt
import numpy as np


# cx 20250611
# 可视化相机轨迹 c2ws ，用于 train.py
def visualize_camera_path(c2ws, scene_bbox, save_path="camera_path.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取相机位置
    cam_positions = np.array([c2w[:3, 3] for c2w in c2ws])
    
    # 绘制相机路径
    ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
            'b-', linewidth=1, label='Camera Path')
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
               c='r', s=20, marker='o', label='Camera Positions')
    
    # 计算边界框顶点
    bbox = np.array(scene_bbox.cpu().numpy())
    vertices = np.array([
        [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[1, 2]]
    ])
    
    # 绘制边界框顶点
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='g', s=50, marker='o', label='BBox Vertices')
    
    # 连接边界框顶点
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'g-', linewidth=1)
    
    
    # 添加世界坐标原点 (重要!)
    ax.scatter([0], [0], [0], c='k', s=150, marker='*', label='World Origin')
    
    # 添加场景边界框中心点
    bbox_center = np.mean(vertices, axis=0)
    ax.scatter([bbox_center[0]], [bbox_center[1]], [bbox_center[2]], 
               c='m', s=100, marker='X', label='Scene Center')
    
    # 在每个相机位置添加朝向指示 (新添加)
    for i in range(0, len(c2ws), max(1, len(c2ws)//10)):  # 每隔10帧画一个
        c2w = c2ws[i]
        cam_pos = c2w[:3, 3]
        
        # 绘制朝向向量 (c2w矩阵的第三列是相机的朝向方向)
        view_dir = c2w[:3, 2] * 0.5  # 缩小的方向向量
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
                  view_dir[0], view_dir[1], view_dir[2],
                  color='purple', length=1, arrow_length_ratio=0.3, 
                  label='Camera View Direction' if i == 0 else None)
    
    # 添加坐标轴标签和网格
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # 添加坐标系原点指示器
    axis_scale = np.max(np.abs(vertices)) * 0.3  # 根据场景大小缩放
    ax.quiver(0, 0, 0, axis_scale, 0, 0, color='r', label='X Axis', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_scale, 0, color='g', label='Y Axis', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_scale, color='b', label='Z Axis', arrow_length_ratio=0.1)
    
    # 调整视角使原点更明显
    ax.view_init(elev=30, azim=45)
    
    # 设置图例位置并保存
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('Camera Path Visualization')
    ax.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path)
    plt.close(fig)  # 避免显示图像时卡住
    return save_path

