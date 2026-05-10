import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从data.txt读取数据
def load_data_from_file(filename='data.txt'):
    records = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split(',')
            # 格式: 时间,算法,攻击方式,强度1,强度2,奖励
            if len(parts) >= 6:
                algorithm = parts[1]
                attack_method = parts[2]
                intensity1 = parts[3]
                intensity2 = parts[4]
                reward = float(parts[5])
                
                # 创建攻击名称（包含强度信息）
                if attack_method == 'none':
                    attack_name = 'none'
                else:
                    # 对于有攻击的情况，组合攻击方法和强度
                    attack_name = f"{attack_method}_{intensity1}"
                    if float(intensity2) != 0:
                        attack_name += f"_{intensity2}"
                
                records.append({
                    'algorithm': algorithm,
                    'attack': attack_name,
                    'reward': reward
                })
    return pd.DataFrame(records)

# 加载数据
df = load_data_from_file('data.txt')

# 显示原始数据
print("原始数据:")
print(df)
print("\n")

# 获取所有唯一的算法和攻击方式
algorithms = df['algorithm'].unique()
attacks = df['attack'].unique()

print(f"算法列表: {algorithms}")
print(f"攻击方式列表: {attacks}")
print("\n")

# 创建透视表
pivot_table = df.pivot(index='algorithm', columns='attack', values='reward')

# 重新排序列，让'none'在第一列
if 'none' in attacks:
    cols = ['none'] + [col for col in attacks if col != 'none']
    pivot_table = pivot_table[cols]

# 打印表格（文本形式）
print("="*80)
print("奖励值表格 (Reward Table)")
print("="*80)
print(pivot_table.round(4))
print("="*80)
print("\n")

# 1. 绘制热力图
fig, ax = plt.subplots(figsize=(10, 6))

# 创建热力图
im = ax.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto', vmin=-80, vmax=-45)

# 设置坐标轴
ax.set_xticks(np.arange(len(pivot_table.columns)))
ax.set_yticks(np.arange(len(pivot_table.index)))
ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(pivot_table.index, fontsize=10)

# 在每个单元格中添加数值
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Reward", rotation=-90, va="bottom", fontsize=10)

# 标题和标签
ax.set_xlabel("Attack Methods", fontsize=12, fontweight='bold')
ax.set_ylabel("Algorithms", fontsize=12, fontweight='bold')
ax.set_title("Reward Comparison: Algorithms vs Attack Methods", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('reward_table_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 热力图已保存为: reward_table_heatmap.png")

# 2. 绘制分组柱状图
fig2, ax2 = plt.subplots(figsize=(12, 6))

# 设置柱状图参数
x = np.arange(len(algorithms))
width = 0.15
colors = plt.cm.tab10(np.linspace(0, 1, len(attacks)))

for i, attack in enumerate(attacks):
    rewards = [pivot_table.loc[algo, attack] for algo in algorithms]
    offset = (i - len(attacks)/2) * width + width/2
    bars = ax2.bar(x + offset, rewards, width, label=attack, color=colors[i])
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Algorithms', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
ax2.set_title('Reward Comparison: Algorithms vs Attack Methods', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms, rotation=15, ha='right')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(bottom=min(pivot_table.min()) - 5, top=max(pivot_table.max()) + 5)

plt.tight_layout()
plt.savefig('reward_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✓ 柱状图已保存为: reward_comparison_bars.png")

# 3. 绘制折线图（更清晰展示每个算法在不同攻击下的表现）
fig3, ax3 = plt.subplots(figsize=(10, 6))

markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']
colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))

for idx, algo in enumerate(algorithms):
    rewards = [pivot_table.loc[algo, attack] for attack in attacks]
    ax3.plot(attacks, rewards, marker=markers[idx % len(markers)], 
            color=colors[idx], linewidth=2, markersize=8, label=algo)

ax3.set_xlabel('Attack Methods', fontsize=12, fontweight='bold')
ax3.set_ylabel('Reward', fontsize=12, fontweight='bold')
ax3.set_title('Algorithm Performance Under Different Attacks', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('reward_trend_lines.png', dpi=300, bbox_inches='tight')
print("✓ 趋势图已保存为: reward_trend_lines.png")

plt.show()

# 4. 统计摘要
print("\n" + "="*80)
print("统计摘要")
print("="*80)

print("\n按攻击方式分析:")
print("-"*40)
for attack in attacks:
    print(f"\n攻击方法: {attack}")
    attack_data = pivot_table[attack]
    print(f"  平均奖励: {attack_data.mean():.4f}")
    print(f"  标准差: {attack_data.std():.4f}")
    print(f"  最佳算法: {attack_data.idxmax()} ({attack_data.max():.4f})")
    print(f"  最差算法: {attack_data.idxmin()} ({attack_data.min():.4f})")

print("\n" + "-"*40)
print("\n按算法分析:")
print("-"*40)
for algo in algorithms:
    print(f"\n算法: {algo}")
    algo_data = pivot_table.loc[algo]
    print(f"  平均奖励: {algo_data.mean():.4f}")
    print(f"  标准差: {algo_data.std():.4f}")
    print(f"  无攻击时奖励: {algo_data['none']:.4f}")
    
    # 计算性能下降
    print(f"  抗攻击性能下降:")
    for attack in attacks:
        if attack != 'none':
            degradation = algo_data['none'] - algo_data[attack]
            degradation_pct = (degradation / abs(algo_data['none'])) * 100
            print(f"    攻击 {attack}: {algo_data[attack]:.4f} (下降 {degradation:.4f} / {degradation_pct:.1f}%)")

# 5. 导出表格为CSV
pivot_table.to_csv('reward_table.csv')
print("\n✓ 奖励表格已导出为: reward_table.csv")

# 6. 找出最佳组合
print("\n" + "="*80)
print("关键发现")
print("="*80)

# 整体最佳表现
best_overall = pivot_table.max().max()
best_algo = pivot_table.stack()[pivot_table.stack() == best_overall].index[0]
print(f"\n🏆 整体最佳表现: {best_algo[0]} 在 {best_algo[1]} 下获得 {best_overall:.4f}")

# 最鲁棒的算法（平均奖励最高）
most_robust_algo = pivot_table.mean(axis=1).idxmax()
print(f"🛡️ 最鲁棒算法: {most_robust_algo} (平均奖励 {pivot_table.mean(axis=1).max():.4f})")

# 受攻击影响最小的算法
if 'none' in attacks:
    degradation_rates = {}
    for algo in algorithms:
        base_reward = pivot_table.loc[algo, 'none']
        attacked_rewards = [pivot_table.loc[algo, attack] for attack in attacks if attack != 'none']
        avg_degradation = base_reward - np.mean(attacked_rewards)
        degradation_rates[algo] = avg_degradation / abs(base_reward) * 100
    
    most_robust_to_attack = min(degradation_rates, key=degradation_rates.get)
    print(f"💪 最抗攻击算法: {most_robust_to_attack} (平均性能下降 {degradation_rates[most_robust_to_attack]:.1f}%)")

# 最致命的攻击
if 'none' in attacks:
    attack_impact = {}
    for attack in attacks:
        if attack != 'none':
            base_rewards = pivot_table['none']
            attacked_rewards = pivot_table[attack]
            avg_impact = (base_rewards - attacked_rewards).mean()
            attack_impact[attack] = avg_impact
    
    most_deadly_attack = max(attack_impact, key=attack_impact.get)
    print(f"⚡ 最致命攻击: {most_deadly_attack} (平均造成 {attack_impact[most_deadly_attack]:.4f} 奖励下降)")