import os
import re
from tensorboard.backend.event_processing import event_accumulator
import os
import datetime

results_path = 'results'
envs = [
    # 'rendezvous', 
    # 'navigation', 
    'pursuit', 
    # 'smacv2'
]
settings = [
    # 'hepnlocal-direct', 
    # 'hepnlocal-direct-tac', 
    # 'hepnlocal-directacc', 
    # 'protoss_5_vs_5', 
    # 'terran_5_vs_5', 
    # 'zerg_5_vs_5'
]
# algo = ''
names = [
    'test',
    # 'tnnls_10_0.25',
    # 'tnnls_10_0.3',
    # 'tnnls_10_0.35',
    # 'tnnls_10_0.4',
]



for env in envs:
    env_path = os.path.join(results_path, env)
    # for setting in settings:
    for setting in os.listdir(env_path):
        setting_path = os.path.join(env_path, setting)
        if os.path.exists(setting_path):
            for algo in os.listdir(setting_path):
                algo_path = os.path.join(setting_path, algo)
                # for name in names:
                for name in os.listdir(algo_path):
                    seed_dir = os.path.join(algo_path, name)
                    if os.path.exists(seed_dir):
                        for seed in os.listdir(seed_dir):
                            log_file_path = os.path.join(seed_dir, seed)
                            match = re.search(r'results/([^/]+)/([^/]+)/([^/]+)/([^/]+)', log_file_path)
                            task, setting, algorithm, name = match.groups()
                            seed_number = int(re.search(r'seed-([^-]+)', log_file_path).group(1))

                            if task in ['smacv2']:
                                log_file_path += '/logs/eval_win_rate/eval_win_rate/'
                            else:
                                log_file_path += '/logs/train_episode_rewards/aver_rewards/'

                            for filename in os.listdir(log_file_path):
                                log_file = log_file_path + filename

                            ea = event_accumulator.EventAccumulator(log_file)
                            ea.Reload()

                            # 以 'train/loss' 为例读取 scalar 信息（你也可以选择其他 tag）
                            tags = ea.Tags()['scalars']  # 查看所有 tag
                            scalars = ea.Scalars(tags[0])  # 假设选择第一个 scalar

                            # 获取时间戳信息
                            start_time = scalars[0].wall_time
                            end_time = scalars[-1].wall_time

                            duration_seconds = end_time - start_time
                            duration_hours = duration_seconds / 3600
                            # duration_str = str(datetime.timedelta(seconds=int(duration_seconds)))

                            # print(f"实验时长为：{duration_str}")
                            print(env, setting, algo, name, round(duration_hours, 2))
