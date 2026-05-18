for SEED in 24
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --algo gcn_mappo \
        --exp_name test \
        --env navigation \
        --obs_mode hepn_local \
        --local_mode False \
        --nr_agents 10 \
        --world_size 100 \
        --obs_radius 141.4 \
        --structural_entropy False \
        --torus True \
        --dynamics direct \
        --env_num1 0.0\
        --env_num2 5.0\
        --use_recurrent_policy False \
        --seed $SEED \
        --num_env_steps 2000000 \
        --lr 0.0005 \
        --critic_lr 0.0005 \
        --use_clipped_value_loss True \
        --ppo_epoch 5 \
        --critic_epoch 5 \
        --n_rollout_threads 10 \
        --use_eval False \
        --log_dir "./results"
done