'''下载训练数据'''
# import tensorflow as tf
# import os
# import re
# import csv

# log_file_path = '/mnt/data/ssd/tyk/EGH-MARL/results_eswa/smacv2/protoss_5_vs_5/hama_mappo/test/seed-00006-2026-02-23-01-49-30'

# match = re.search(r'results_eswa/([^/]+)/([^/]+)/([^/]+)/([^/]+)', log_file_path)
# task, setting, algorithm, name = match.groups()
# seed_number = int(re.search(r'seed-([^-]+)', log_file_path).group(1))

# if task in ['smacv2']:
#     log_file_path += '/logs/eval_win_rate/eval_win_rate/'
# else:
#     log_file_path += '/logs/train_episode_rewards/aver_rewards/'

# for filename in os.listdir(log_file_path):
#     log_file = log_file_path + filename

# if task in ['mujoco3d', 'smacv2']:
#     csv_file_dir = os.path.join('csv_data_eswa', setting, name, algorithm)
# else:
#     csv_file_dir = os.path.join('csv_data_eswa', task, name, algorithm)
# os.makedirs(csv_file_dir, exist_ok=True)

# csv_file_path = os.path.join(csv_file_dir, 'seed_' + str(seed_number)) + '.csv'

# with open(csv_file_path, mode='w', newline='') as file:
#     writre = csv.writer(file)
#     writre.writerow(['Step', 'Episode Rewards'])
#     for e in tf.compat.v1.train.summary_iterator(log_file):
#         for v in e.summary.value:
#             writre.writerow([e.step, v.simple_value])

'''计算训练时间'''
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime
import re

log_file_path = '/mnt/data/ssd/tyk/EGH-MARL/results_eswa/smacv2/zerg_5_vs_5/hmf_mappo/test/seed-00004-2026-02-21-15-05-26'
match = re.search(r'results_eswa/([^/]+)/([^/]+)/([^/]+)/([^/]+)', log_file_path)
task, setting, algorithm, name = match.groups()
if task in ['smacv2']:
    log_file_path += '/logs/eval_win_rate/eval_win_rate/'
else:
    log_file_path += '/logs/train_episode_rewards/aver_rewards/'

ea = event_accumulator.EventAccumulator(log_file_path)
ea.Reload()

# 先看有哪些标量tag
tags = ea.Tags().get("scalars", [])
print("scalar tags:", tags)

# 选一个持续记录的tag（例如 rewards 或 win_rate）
tag = tags[0]
events = ea.Scalars(tag)

start_ts = events[0].wall_time
end_ts = events[-1].wall_time
duration_sec = end_ts - start_ts

print("start:", datetime.fromtimestamp(start_ts))
print("end  :", datetime.fromtimestamp(end_ts))
print(f"duration: {duration_sec:.1f}s ({duration_sec/3600:.2f}h)")