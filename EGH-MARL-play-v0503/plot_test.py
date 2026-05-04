import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 读取数据文件
data = []
with open('test_log.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # 跳过空行
            parts = line.split()
            #print(parts)
            if len(parts) == 4:  # 确保有4个字段
                # 更安全地解析每一部分
                x = float(parts[0])
                y = float(parts[1])
                ax = float(parts[2])
                ay = float(parts[3])
                #print(x,y,ax,ay)
                data.append([x, y, ax, ay])

# 转换为numpy数组
data = np.array(data)

# 检查数据形状
print(f"数据形状: {data.shape}")
print(f"数据前5行:\n{data[:5]}")

# 提取各列
x = data[:, 0]
y = data[:, 1]
ax = data[:, 2]
ay = data[:, 3]

print(f"\nX范围: [{x.min():.3f}, {x.max():.3f}]")
print(f"Y范围: [{y.min():.3f}, {y.max():.3f}]")
print(f"ax范围: [{ax.min():.3f}, {ax.max():.3f}]")
print(f"ay范围: [{ay.min():.3f}, {ay.max():.3f}]")

# 创建网格用于插值（如果需要更密集的显示）
# 使用原始数据的范围
x_grid = np.linspace(x.min(), x.max(), 100)
y_grid = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)

# 对ax和ay进行插值
Z_ax = griddata((x, y), ax, (X, Y), method='cubic', fill_value=np.nan)
Z_ay = griddata((x, y), ay, (X, Y), method='cubic', fill_value=np.nan)

# 创建图形 - 深度图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制第一张深度图像 (ax)
im1 = ax1.contourf(X, Y, Z_ax, levels=20, cmap='viridis')
plt.colorbar(im1, ax=ax1, label='ax depth')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Depth Map for ax')
ax1.set_aspect('equal')

# 绘制第二张深度图像 (ay)
im2 = ax2.contourf(X, Y, Z_ay, levels=20, cmap='plasma')
plt.colorbar(im2, ax=ax2, label='ay depth')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Depth Map for ay')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('depth_maps.png', dpi=300, bbox_inches='tight')
print("深度图已保存为 depth_maps.png")
plt.close()

# 创建图形 - 散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = ax1.scatter(x, y, c=ax, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter1, ax=ax1, label='ax depth')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Scatter Plot for ax (Original Data)')
ax1.set_aspect('equal')

scatter2 = ax2.scatter(x, y, c=ay, cmap='plasma', s=100, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter2, ax=ax2, label='ay depth')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Scatter Plot for ay (Original Data)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
print("散点图已保存为 scatter_plots.png")
plt.close()

print("所有图像已保存完成！")