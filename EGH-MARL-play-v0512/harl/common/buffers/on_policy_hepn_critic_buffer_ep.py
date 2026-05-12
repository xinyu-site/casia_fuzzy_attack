from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
import numpy as np
import copy
import torch

class OnPolicyHepnCriticBufferEP(OnPolicyCriticBufferEP):
    def __init__(self, args, share_obs_space, use_history=False, windows_size=1):
        super(OnPolicyHepnCriticBufferEP, self).__init__(args, share_obs_space, use_history, windows_size)
        self.graphs = np.empty((self.episode_length + 1, self.n_rollout_threads, ), dtype=object)
    
    def insert(
        self, share_obs, layer_data, rnn_states_critic, value_preds, rewards, masks, bad_masks
    ):
        """Insert data into buffer."""
        share_obs = share_obs.squeeze()
        self.share_obs[self.step + 1] = share_obs.copy()
        self.graphs[self.step + 1] = copy.deepcopy(layer_data)
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def feed_forward_generator_critic(
        self, critic_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for critic that uses MLP network.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            mini_batch_size: (int) Size of mini batch for critic.
        """

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) = {n_rollout_threads * episode_length} "
                f"is required to be greater than or equal to the number of critic mini batches ({critic_num_mini_batch})."
            )
            mini_batch_size = batch_size // critic_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (episode_length * n_rollout_threads, *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )  # actually not used, just for consistency
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        graphs = self.graphs[:-1].reshape(-1, )

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            layer_data = graphs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, layer_data, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.graphs[0] = copy.deepcopy(self.graphs[-1])
