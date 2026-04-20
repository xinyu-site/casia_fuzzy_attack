import tensorflow as tf
import os
import re
import csv

# ===================== 核心配置：只需要改这一个根路径 =====================
ROOT_DIR = '/home/yuxin/fuzzymarl/EGH-MARL-new/results/navigation/hepnlocal-direct'

def convert_tb_to_csv(seed_dir):
    """
    处理单个种子文件夹：读取日志 → 生成CSV
    """
    # 1. 从路径中提取关键信息
    # 路径格式：ROOT_DIR/算法名/任务名/种子文件夹
    path_parts = seed_dir.strip('/').split('/')
    # 从后往前取：seed文件夹 → 任务名 → 算法名
    seed_folder_name = path_parts[-1]
    task = path_parts[-2]
    algorithm = path_parts[-3]
    
    # 提取种子数字
    seed_match = re.search(r'seed-(\d+)', seed_folder_name)
    if not seed_match:
        print(f"⚠️  跳过非种子文件夹: {seed_dir}")
        return
    seed_number = int(seed_match.group(1))
    
    # 2. 定位日志文件路径
    if task in ['smacv2']:
        log_sub_dir = 'logs/eval_win_rate/eval_win_rate/'
    else:
        log_sub_dir = 'logs/train_episode_rewards/aver_rewards/'
    
    log_full_dir = os.path.join(seed_dir, log_sub_dir)
    
    # 检查日志目录是否存在
    if not os.path.isdir(log_full_dir):
        print(f"❌ 日志目录不存在: {log_full_dir}")
        return
    
    # 找到唯一的日志文件
    log_files = [f for f in os.listdir(log_full_dir) if os.path.isfile(os.path.join(log_full_dir, f))]
    if not log_files:
        print(f"❌ 无日志文件: {log_full_dir}")
        return
    log_file_path = os.path.join(log_full_dir, log_files[0])

    # 3. 创建CSV输出目录
    csv_file_dir = os.path.join('trans_csv_data', task, algorithm)
    os.makedirs(csv_file_dir, exist_ok=True)
    csv_file_path = os.path.join(csv_file_dir, f'seed_{seed_number}.csv')

    # 4. 读取TensorBoard日志并写入CSV
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Episode Rewards'])
            
            for e in tf.compat.v1.train.summary_iterator(log_file_path):
                for v in e.summary.value:
                    writer.writerow([e.step, v.simple_value])
        
        print(f"✅ 转换完成: {csv_file_path}")
    except Exception as e:
        print(f"❌ 转换失败 {seed_dir}: {str(e)}")

def batch_convert_all(root_dir):
    """
    批量遍历：算法文件夹 → 任务文件夹 → 种子文件夹
    """
    print(f"🚀 开始批量转换，根目录: {root_dir}\n")
    
    # 遍历所有算法文件夹
    for algo_name in os.listdir(root_dir):
        algo_path = os.path.join(root_dir, algo_name)
        if not os.path.isdir(algo_path):
            continue
        
        # 遍历算法下的所有任务文件夹
        for task_name in os.listdir(algo_path):
            task_path = os.path.join(algo_path, task_name)
            if not os.path.isdir(task_path):
                continue
            
            # 遍历任务下的所有种子文件夹
            for seed_folder in os.listdir(task_path):
                seed_path = os.path.join(task_path, seed_folder)
                if os.path.isdir(seed_path) and 'seed-' in seed_folder:
                    convert_tb_to_csv(seed_path)

if __name__ == '__main__':
    batch_convert_all(ROOT_DIR)
    print("\n🎉 所有文件批量转换完成！")