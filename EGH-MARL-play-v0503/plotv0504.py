import matplotlib.pyplot as plt
from collections import defaultdict

# 读取数据文件
data = defaultdict(list)  # key: 算法名, value: list of (攻击强度, 平均奖励)

with open('eval_result_pursuit.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        # 列索引：0-时间戳, 1-算法, 2-环境, 3-攻击强度, 4-额外参数, 5-平均奖励
        algo = parts[1]
        attack = float(parts[3])
        reward = float(parts[5])
        data[algo].append((attack, reward))

# 按攻击强度排序每个算法的数据点
for algo in data:
    data[algo].sort(key=lambda x: x[0])

# 绘制折线图
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 不同标记
colors = plt.cm.tab10.colors  # 使用tab10颜色映射

for i, (algo, points) in enumerate(data.items()):
    attacks = [p[0] for p in points]
    rewards = [p[1] for p in points]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    plt.plot(attacks, rewards, marker=marker, color=color, label=algo, linewidth=2, markersize=8)

plt.xlabel('Attack Strength', fontsize=14)
plt.ylabel('Average Reward', fontsize=14)
plt.title('Algorithm Performance under Different Attack Strengths', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('algorithm_performance.png', dpi=150)  # 保存图片
plt.show()