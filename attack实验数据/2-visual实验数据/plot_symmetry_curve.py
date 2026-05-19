import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据（请修改为你的文件路径）
data = np.loadtxt('model_test_log.txt')  # 替换为你的文件路径

rad = data[:, 0]      # 弧度
z1 = data[:, 1]       # 第二列
z2 = data[:, 2]       # 第三列

# 计算 x, y
x = np.cos(rad)
y = np.sin(rad)

# 创建三维图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制两条曲线
ax.plot(x, y, z1, label='Curve 1 (Column 2)', color='blue', linewidth=2)
ax.plot(x, y, z2, label='Curve 2 (Column 3)', color='red', linewidth=2)

# 设置标签
ax.set_xlabel('X = cos(rad)', fontsize=12)
ax.set_ylabel('Y = sin(rad)', fontsize=12)
ax.set_zlabel('Z Value', fontsize=12)
ax.set_title('3D Curves on Cylindrical Surface', fontsize=14)

# 设置 z 轴范围
ax.set_zlim(-4, 4)

# 添加图例
ax.legend(fontsize=10)

# 设置坐标轴等比例
ax.set_box_aspect([1, 1, 0.8])

# 保存图片（dpi 可调整）
plt.savefig('3d_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('3d_curves.pdf', bbox_inches='tight')  # 同时保存 PDF 格式

print("图片已保存为 '3d_curves.png' 和 '3d_curves.pdf'")

# 如果你想关闭图形窗口而不显示，可以注释掉下一行
# plt.show()