import matplotlib.pyplot as plt
import pandas as pd

# 读取你的数据文件
df = pd.read_csv('eval_result.txt', header=None,
                 names=['time', 'algo', 'attack', 'strength', 'unused', 'reward'])

# 拼接攻击方式+强度（不同参数=不同横轴）
df['x_label'] = df['attack'] + '_' + df['strength'].astype(str)
x_order = df['x_label'].unique()

# 整理数据 - 先聚合重复值
pivot = df.groupby(['algo', 'x_label'])['reward'].mean().unstack()
pivot = pivot.reindex(columns=x_order)

# --------------------- 核心：去掉中文，不报错 ---------------------
plt.rcParams['font.family'] = ['DejaVu Sans']  # 通用字体
plt.figure(figsize=(18, 7))

# 画4条算法折线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728' ,'#9467bd' , '#8c564b']
for i, algo in enumerate(pivot.index):
    plt.plot(pivot.columns, pivot.loc[algo], marker='o', linewidth=2.5, label=algo, color=colors[i])

# 全英文标签（不报错）
plt.xlabel('Attack_Type_Strength', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('Average Reward under Different Attacks and Strengths', fontsize=14)
plt.legend(fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存高清图片 + 显示
plt.savefig('attack_reward_plot.png', dpi=300, bbox_inches='tight')
plt.show()