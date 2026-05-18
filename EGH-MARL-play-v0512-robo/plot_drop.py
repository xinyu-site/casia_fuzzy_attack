import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 读取数据
def parse_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split(',')
                timestamp = parts[0]
                algorithm = parts[1]
                attack_type = parts[2]
                noise_1 = float(parts[3])
                noise_2 = float(parts[4])
                avg_reward = float(parts[5])
                data.append({
                    'timestamp': timestamp,
                    'algorithm': algorithm,
                    'attack_type': attack_type,
                    'noise_intensity': (noise_1, noise_2),
                    'noise_str': f"({noise_1}, {noise_2})",
                    'avg_reward': avg_reward
                })
    return data

# 计算性能下降率
def calculate_drop_rate(baseline_reward, attacked_reward):
    if baseline_reward == 0:
        return 0
    drop_rate = ((baseline_reward - attacked_reward) / abs(baseline_reward)) * 100
    return drop_rate

# 生成漂亮的HTML表格
def generate_html_table(result_df, algorithms, attack_levels):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>算法性能对比分析报告</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 14px;
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px;
                text-align: center;
                font-weight: bold;
                border: 1px solid #ddd;
            }
            td {
                padding: 10px;
                text-align: center;
                border: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            tr:hover {
                background-color: #e3f2fd;
                transition: 0.3s;
            }
            .algo-name {
                font-weight: bold;
                background-color: #e8eaf6;
                color: #283593;
            }
            .drop-positive {
                color: #e74c3c;
                font-weight: bold;
            }
            .drop-negative {
                color: #27ae60;
                font-weight: bold;
            }
            .reward-high {
                color: #27ae60;
                font-weight: bold;
            }
            .caption {
                margin-top: 20px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 强化学习算法对抗攻击性能分析</h1>
            <div class="subtitle">不同攻击强度下的平均奖励与性能下降率对比</div>
            <table>
                <thead>
                    <tr>
                        <th>算法</th>
    """
    
    # 添加列标题
    for level in attack_levels:
        html += f"<th colspan='2'>{level}</th>"
    html += "</tr><tr><th></th>"
    
    for level in attack_levels:
        html += "<th>平均奖励</th><th>下降率 (%)</th>"
    html += "</tr></thead><tbody>"
    
    # 添加数据行
    for idx, row in result_df.iterrows():
        html += "<tr>"
        html += f"<td class='algo-name'>{row['Algorithm']}</td>"
        
        for i, level in enumerate(attack_levels):
            reward_key = f'reward_{level}'
            drop_key = f'drop_{level}'
            
            if reward_key in row:
                reward = row[reward_key]
                html += f"<td>{reward:.2f}</td>"
                
                if drop_key in row:
                    drop_rate = row[drop_key]
                    drop_class = "drop-positive" if drop_rate > 0 else "drop-negative" if drop_rate < 0 else ""
                    if i == 0:
                        html += "<td>-</td>"
                    else:
                        html += f"<td class='{drop_class}'>{drop_rate:+.2f}%</td>"
                else:
                    html += "<td>-</td>"
            else:
                html += "<td>-</td><td>-</td>"
        
        html += "</tr>"
    
    html += """
            </tbody>
        </table>
        <div class="caption">
            <strong>📊 说明：</strong><br>
            • 下降率为正值表示性能下降，负值表示性能提升<br>
            • 红色数字表示性能下降，绿色数字表示性能提升<br>
            • 基线为无攻击状态<br>
            • 下降率计算公式: (基线奖励 - 攻击后奖励) / |基线奖励| × 100%
        </div>
        </div>
    </body>
    </html>
    """
    return html

# 绘制drop_rate柱状图
def plot_drop_rates(df, algorithms, attack_levels):
    """绘制各种攻击强度下的drop_rate对比图"""
    baseline_intensity = attack_levels[0]
    
    # 为每个非基线的攻击强度准备数据
    drop_rates_data = {}
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for intensity in attack_levels[1:]:  # 跳过基线
        drop_rates = []
        for algo in algorithms:
            algo_data = df[df['algorithm'] == algo]
            baseline = algo_data[algo_data['noise_intensity'] == baseline_intensity]['avg_reward'].values[0]
            attacked = algo_data[algo_data['noise_intensity'] == intensity]['avg_reward'].values[0]
            drop_rate = calculate_drop_rate(baseline, attacked)
            drop_rates.append(drop_rate)
        drop_rates_data[f'{intensity}'] = drop_rates
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(algorithms))
    width = 0.8 / len(drop_rates_data)
    
    # 绘制分组柱状图
    for i, (attack_name, drop_rates) in enumerate(drop_rates_data.items()):
        offset = (i - len(drop_rates_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, drop_rates, width, label=attack_name, 
                     color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在柱子上添加数值标签
        for bar, dr in zip(bars, drop_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (1 if dr >= 0 else -3),
                   f'{dr:.1f}%', ha='center', va='bottom' if dr >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    # 设置图表属性
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drop Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Drop Rate Under Different Attack Intensities\n(Positive = Performance Degradation, Negative = Improvement)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 设置y轴范围（留出一些空间给数值标签）
    all_drop_rates = [dr for drops in drop_rates_data.values() for dr in drops]
    y_min = min(all_drop_rates) - 10 if min(all_drop_rates) < 0 else -10
    y_max = max(all_drop_rates) + 10
    ax.set_ylim(y_min, y_max)
    
    # 添加水平参考线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('drop_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Drop rate对比图已保存: drop_rate_comparison.png")

def main():
    # 读取数据
    data = parse_data('pursuit.txt')
    df = pd.DataFrame(data)
    
    # 获取所有算法和攻击强度
    algorithms = df['algorithm'].unique()
    attack_levels = sorted(df['noise_intensity'].unique(), key=lambda x: (x[0], x[1]))
    
    print(f"发现 {len(algorithms)} 个算法: {list(algorithms)}")
    print(f"发现 {len(attack_levels)} 个攻击强度: {attack_levels}")
    
    # 创建结果表格
    result_rows = []
    
    for algorithm in algorithms:
        algo_data = df[df['algorithm'] == algorithm]
        row = {'Algorithm': algorithm}
        
        for intensity in attack_levels:
            intensity_data = algo_data[algo_data['noise_intensity'] == intensity]
            if len(intensity_data) > 0:
                reward = intensity_data['avg_reward'].values[0]
                row[f'reward_{intensity}'] = reward
        
        # 计算下降率
        baseline_intensity = attack_levels[0]  # 第一个作为基线
        baseline_reward = row.get(f'reward_{baseline_intensity}')
        
        if baseline_reward is not None:
            for intensity in attack_levels[1:]:  # 跳过基线
                attacked_reward = row.get(f'reward_{intensity}')
                if attacked_reward is not None:
                    drop_rate = calculate_drop_rate(baseline_reward, attacked_reward)
                    row[f'drop_{intensity}'] = drop_rate
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    
    # 保存CSV文件
    # 重新组织CSV格式，使其更易读
    csv_data = []
    for _, row in result_df.iterrows():
        csv_row = {'Algorithm': row['Algorithm']}
        for intensity in attack_levels:
            reward_key = f'reward_{intensity}'
            drop_key = f'drop_{intensity}'
            if reward_key in row:
                csv_row[f'Reward_{intensity}'] = row[reward_key]
            if drop_key in row:
                csv_row[f'Drop_Rate_{intensity}(%)'] = row[drop_key]
        csv_data.append(csv_row)
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv('algorithm_performance.csv', index=False, encoding='utf-8-sig')
    print("✅ CSV文件已保存: algorithm_performance.csv")
    
    # 创建强度标签用于显示
    intensity_labels = {level: f"({level[0]}, {level[1]})" for level in attack_levels}
    
    # 打印控制台表格（美化）
    print("\n" + "="*120)
    print(" "*40 + "算法性能对比分析报告")
    print("="*120)
    
    # 构建控制台表格数据
    console_table = []
    for _, row in result_df.iterrows():
        console_row = [row['Algorithm']]
        for level in attack_levels:
            reward_key = f'reward_{level}'
            drop_key = f'drop_{level}'
            if reward_key in row:
                console_row.append(f"{row[reward_key]:.2f}")
                if drop_key in row:
                    console_row.append(f"{row[drop_key]:+.2f}%")
                else:
                    console_row.append("-")
            else:
                console_row.append("-")
                console_row.append("-")
        console_table.append(console_row)
    
    # 创建表头
    headers = ["Algorithm"]
    for level in attack_levels:
        headers.append(f"{intensity_labels[level]}\n(Reward)")
        if level != attack_levels[0]:
            headers.append(f"{intensity_labels[level]}\n(Drop Rate)")
        else:
            headers.append("")
    
    print(tabulate(console_table, headers=headers, tablefmt="grid", 
                   numalign="center", stralign="center"))
    
    # 生成HTML表格
    html_table = generate_html_table(result_df, algorithms, 
                                     [f"{intensity_labels[l]}" for l in attack_levels])
    with open('performance_report.html', 'w', encoding='utf-8') as f:
        f.write(html_table)
    print("\n✅ HTML报告已生成: performance_report.html")
    
    # 生成Drop Rate柱状图
    plot_drop_rates(df, algorithms, attack_levels)
    
    # 打印详细统计信息
    print("\n" + "="*120)
    print("详细统计信息")
    print("="*120)
    
    baseline_intensity = attack_levels[0]
    for algorithm in algorithms:
        print(f"\n📊 算法: {algorithm}")
        algo_data = df[df['algorithm'] == algorithm]
        
        for intensity in attack_levels:
            reward = algo_data[algo_data['noise_intensity'] == intensity]['avg_reward'].values[0]
            print(f"   • 攻击强度 {intensity_labels[intensity]}: 平均奖励 = {reward:.4f}")
        
        baseline_reward = algo_data[algo_data['noise_intensity'] == baseline_intensity]['avg_reward'].values[0]
        print(f"   📈 性能变化:")
        for intensity in attack_levels[1:]:
            attacked_reward = algo_data[algo_data['noise_intensity'] == intensity]['avg_reward'].values[0]
            drop_rate = calculate_drop_rate(baseline_reward, attacked_reward)
            if drop_rate > 0:
                print(f"      → {intensity_labels[intensity]}: 下降 {drop_rate:.2f}% (性能降低 ⚠️)")
            elif drop_rate < 0:
                print(f"      → {intensity_labels[intensity]}: 提升 {abs(drop_rate):.2f}% (性能提升 ✨)")
            else:
                print(f"      → {intensity_labels[intensity]}: 无变化")
    
    print("\n" + "="*120)
    print("✅ 分析完成！生成的文件：")
    print("   • algorithm_performance.csv - CSV数据文件")
    print("   • performance_report.html - HTML格式报告")
    print("   • drop_rate_comparison.png - Drop rate对比柱状图")
    print("="*120)

if __name__ == "__main__":
    main()