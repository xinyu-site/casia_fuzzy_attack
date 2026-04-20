
# python train.py --algo mappo --env smacv2 --exp_name rnn --seed 12 --use_eval True --eval_episodes 50 --map_name terran_5_vs_5 --n_rollout_threads 10
# python train.py --algo mappo --env smacv2 --exp_name rnn --seed 12 --use_eval True --eval_episodes 50 --map_name zerg_5_vs_5 --n_rollout_threads 10
# # python train.py --algo mappo --env smacv2 --exp_name rnn --seed 12 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10

# python train.py --algo mappo --env smacv2 --exp_name rnn --seed 13 --use_eval True --eval_episodes 50 --map_name terran_5_vs_5 --n_rollout_threads 10
# python train.py --algo mappo --env smacv2 --exp_name rnn --seed 13 --use_eval True --eval_episodes 50 --map_name zerg_5_vs_5 --n_rollout_threads 10
# # python train.py --algo mappo --env smacv2 --exp_name rnn --seed 13 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10

# CUDA_VISIBLE_DEVICES=6 python train.py --algo mappo --env smacv2 --exp_name rnn --seed 14 --use_eval True --eval_episodes 50 --map_name terran_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=6 python train.py --algo mappo --env smacv2 --exp_name rnn --seed 14 --use_eval True --eval_episodes 50 --map_name zerg_5_vs_5 --n_rollout_threads 10

# CUDA_VISIBLE_DEVICES=6 python train.py --algo mappo --env smacv2 --exp_name rnn --seed 15 --use_eval True --eval_episodes 50 --map_name terran_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=6 python train.py --algo mappo --env smacv2 --exp_name rnn --seed 15 --use_eval True --eval_episodes 50 --map_name zerg_5_vs_5 --n_rollout_threads 10

# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name test --seed 1 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name test --seed 2 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name test --seed 3 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name test --seed 4 --use_eval True --eval_episodes 50 --map_name protoss_5_vs_5 --n_rollout_threads 10

# CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name test --seed 0 --use_eval True --eval_episodes 40 --map_name protoss_10_vs_10 --n_rollout_threads 10 --num_env_steps 20000000
# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name test --seed 0 --use_eval True --eval_episodes 40 --map_name terran_10_vs_10 --n_rollout_threads 10 --num_env_steps 20000000

# CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase --seed 110 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase --seed 111 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase --seed 112 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=4 python train.py --algo mappo --env smacv2 --exp_name tase --seed 113 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10
# CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase --seed 114 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10

# RNN
CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase_rnn --seed 110 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10 --single_actor False --use_recurrent_policy True
CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase_rnn --seed 111 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10 --single_actor False --use_recurrent_policy True
CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase_rnn --seed 112 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10 --single_actor False --use_recurrent_policy True
CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase_rnn --seed 113 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10 --single_actor False --use_recurrent_policy True
CUDA_VISIBLE_DEVICES=5 python train.py --algo mappo --env smacv2 --exp_name tase_rnn --seed 114 --use_eval True --eval_episodes 40 --map_name protoss_5_vs_5 --n_rollout_threads 10 --single_actor False --use_recurrent_policy True