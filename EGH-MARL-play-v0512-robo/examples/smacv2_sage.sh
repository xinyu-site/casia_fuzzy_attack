# python train.py --algo graphsage_mappo --env smacv2 --exp_name ablation --seed 0 --use_eval True --eval_episodes 40 --map_name terran_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000
# python train.py --algo graphsage_mappo --env smacv2 --exp_name ablation --seed 1 --use_eval True --eval_episodes 40 --map_name terran_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000
# python train.py --algo graphsage_mappo --env smacv2 --exp_name ablation --seed 2 --use_eval True --eval_episodes 40 --map_name terran_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000
# python train.py --algo graphsage_mappo --env smacv2 --exp_name ablation --seed 3 --use_eval True --eval_episodes 40 --map_name terran_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000
CUDA_VISIBLE_DEVICES=1 python train.py --algo graphsage_mappo --env smacv2 --exp_name test --seed 50 --use_eval True --eval_episodes 40 --map_name zerg_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000 --use_res False
CUDA_VISIBLE_DEVICES=1 python train.py --algo graphsage_mappo --env smacv2 --exp_name test --seed 51 --use_eval True --eval_episodes 40 --map_name zerg_5_vs_5 --n_rollout_threads 10 --num_env_steps 10000000
