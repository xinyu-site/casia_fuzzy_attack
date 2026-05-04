python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_left_arm.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 2 --lr 0.0002 --critic_lr 0.0002 --use_eval False
python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_left_arm.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 4 --lr 0.0002 --critic_lr 0.0002 --use_eval False
python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_left_arm.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 5 --lr 0.0002 --critic_lr 0.0002 --use_eval False

# python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_right_leg.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 1 --use_eval False
# python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_8_left_knee.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 2 --use_eval False
# python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_left_arm.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 3 --use_eval False
# python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_9_full.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 4 --use_eval False
# python train.py --algo mappo --env mujoco3d --scenario 3d_humanoids --custom_xml 3d_humanoid_7_lower_arms.xml --exp_name test --n_rollout_threads 10 --local_mode False --num_env_steps 10000000 --seed 5 --use_eval False

# python train.py --algo mappo --env mujoco3d --scenario 3d_walkers --custom_xml 3d_walker_3_left_leg_right_foot.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 6 --num_env_steps 10000000 --use_eval True 
# python train.py --algo mappo --env mujoco3d --scenario 3d_walkers --custom_xml 3d_walker_3_left_leg_right_foot.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 7 --num_env_steps 10000000 --use_eval True 
# python train.py --algo mappo --env mujoco3d --scenario 3d_walkers --custom_xml 3d_walker_3_left_leg_right_foot.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 8 --num_env_steps 10000000 --use_eval True 

# python train.py --algo mappo --env mujoco3d --scenario 3d_cheetahs --custom_xml 3d_cheetah_10_tail_leftbleg.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 6 --num_env_steps 10000000 --use_eval True --single_actor True
# python train.py --algo mappo --env mujoco3d --scenario 3d_cheetahs --custom_xml 3d_cheetah_10_tail_leftbleg.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 7 --num_env_steps 10000000 --use_eval True --single_actor True
# python train.py --algo mappo --env mujoco3d --scenario 3d_cheetahs --custom_xml 3d_cheetah_10_tail_leftbleg.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 8 --num_env_steps 10000000 --use_eval True --single_actor True

# python train.py --algo mappo --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 389 --num_env_steps 10000000 --use_eval True --single_actor True

python train.py --algo mappo --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 9 --num_env_steps 10000000 --use_eval False --share_param False --single_actor False
python train.py --algo mappo --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 10 --num_env_steps 10000000 --use_eval False --share_param False --single_actor False
python train.py --algo mappo --env mujoco3d --scenario 3d_hoppers --custom_xml 3d_hopper_3_shin.xml --exp_name test --n_rollout_threads 10 --local_mode False --seed 11 --num_env_steps 10000000 --use_eval False --share_param False --single_actor False