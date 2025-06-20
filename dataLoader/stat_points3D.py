import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


# cx 20250613 分析点云空间分布并输出关键统计信息
def analyze_point_cloud(points3d_path):
    points = []
    with open(points3d_path, 'r') as f:
        for line in f:
            if line.startswith('#'):    # 跳过注释行
                continue
            parts = line.strip().split()
            if len(parts) >= 4:         # 确保有坐标数据，读取 XYZ 坐标
                points.append(list(map(float, parts[1:4])))
    
    if not points:
        print("Not found any valid data.")
        return
    
    points = np.array(points)
    
    # 计算基本统计量
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    
    print("\n========== stats of 3D points ==========")
    print(f"X axis: {min_vals[0]:.2f} → {max_vals[0]:.2f} (range: {ranges[0]:.2f})")
    print(f"Y axis: {min_vals[1]:.2f} → {max_vals[1]:.2f} (range: {ranges[1]:.2f})")
    print(f"Z axis: {min_vals[2]:.2f} → {max_vals[2]:.2f} (range: {ranges[2]:.2f})")
    print(f"Center: ({np.mean(points[:, 0]):.2f}, {np.mean(points[:, 1]):.2f}, {np.mean(points[:, 2]):.2f})")
    
    # 计算密度分布
    percentile = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) * 100
    x_dist = np.percentile(points[:, 0], percentile)
    y_dist = np.percentile(points[:, 1], percentile)
    z_dist = np.percentile(points[:, 2], percentile)
    
    print("\nPercentiles:")
    print("    \t| 10%  \t| 25%  \t| 50%  \t| 75%  \t| 90%")
    print(f"  X \t| {x_dist[0]:.2f}\t| {x_dist[1]:.2f}\t| {x_dist[2]:.2f}\t| {x_dist[3]:.2f}\t| {x_dist[4]:.2f}")
    print(f"  Y \t| {y_dist[0]:.2f}\t| {y_dist[1]:.2f}\t| {y_dist[2]:.2f}\t| {y_dist[3]:.2f}\t| {y_dist[4]:.2f}")
    print(f"  Z \t| {z_dist[0]:.2f}\t| {z_dist[1]:.2f}\t| {z_dist[2]:.2f}\t| {z_dist[3]:.2f}\t| {z_dist[4]:.2f}")

    
    # 图表可视化
    plt.figure(figsize=(16, 12))
    
    # 三维散点密度图
    ax = plt.subplot(223, projection='3d')
    sc = ax.scatter(points[:,0], points[:,1], points[:,2], 
                   c=points[:,2], cmap='viridis', s=1, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D density')
    plt.colorbar(sc, ax=ax, label='Z(height)')
    
    # X轴直方图
    plt.subplot(221)
    n, bins, patches = plt.hist(points[:,0], bins=100, color='blue', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('number of points')
    plt.title('density of X-axis')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Y轴直方图
    plt.subplot(222)
    plt.hist(points[:,1], bins=100, color='green', alpha=0.7)
    plt.xlabel('Y')
    plt.ylabel('number of points')
    plt.title('density of Y-axis')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Z轴直方图
    plt.subplot(224)
    plt.hist(points[:,2], bins=100, color='red', alpha=0.7)
    plt.xlabel('Z')
    plt.ylabel('number of points')
    plt.title('density of Z-axis')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存分析图表
    plot_path = points3d_path.replace('.txt', '_density_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualization stats of 3D points saved at: {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='stat on 3D points in points3D.txt')
    parser.add_argument('--points3D_path', type=str, help='path to points3D.txt')
    args = parser.parse_args()
    
    analyze_point_cloud(args.points3D_path)
