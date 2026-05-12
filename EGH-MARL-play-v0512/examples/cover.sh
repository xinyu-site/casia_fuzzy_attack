#!/bin/sh
python train.py --algo eghn_mappo --env cover --exp_name ecai_5_v2 --obs_mode global --local_mode True --nr_agents 5 --obs_radius 141.5 --seed 1 --num_env_steps 10000000 --use_recurrent_policy False --torus False
python train.py --algo egnn_mappo --env cover --exp_name ecai_5 --obs_mode global --local_mode True --nr_agents 5 --obs_radius 141.5 --seed 0 --num_env_steps 10000000 --use_recurrent_policy False --torus False
python train.py --algo mappo --env cover --exp_name ecai_5 --obs_mode global --local_mode False --nr_agents 5 --obs_radius 141.5 --seed 0 --num_env_steps 10000000 --use_recurrent_policy False --torus False
python train.py --algo graphsage_mappo --env cover --exp_name ecai_5 --obs_mode global --local_mode False --nr_agents 5 --obs_radius 141.5 --seed 0 --num_env_steps 10000000 --use_recurrent_policy False --torus False
python train.py --algo mappo_data_aug --env cover --exp_name ecai_5 --obs_mode global --local_mode False --nr_agents 5 --obs_radius 141.5 --seed 0 --num_env_steps 10000000 --use_recurrent_policy False --torus False
