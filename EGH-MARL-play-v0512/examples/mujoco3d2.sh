# python train.py --algo mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --seed 0
# python train.py --algo mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --lr 0.0001 --critic_lr 0.0001 --seed 0
python train.py --algo eghnv2_mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --n_cluster 3 --lr 0.0002 --critic_lr 0.0002 --seed 1
# python train.py --algo eghnv2_mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --n_cluster 2 --lr 0.0001 --critic_lr 0.0001 --seed 2
# python train.py --algo eghnv2_mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --n_cluster 2 --lr 0.0001 --critic_lr 0.0001 --seed 3
# python train.py --algo eghnv2_mappo --env mujoco3d --exp_name test --n_rollout_threads 10 --local_mode False --n_cluster 2 --lr 0.0001 --critic_lr 0.0001 --seed 4

# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_lower_arms.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 5 --use_eval False --single_actor True
# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_lower_arms.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 6 --use_eval False --single_actor True
# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_lower_arms.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 7 --use_eval False --single_actor True


# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 5 --use_eval False --single_actor True
# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 6 --use_eval False --single_actor True
# python train.py --algo mappo_data_aug --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 7 --use_eval False --single_actor True
