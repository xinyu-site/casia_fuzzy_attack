import copy
import numpy as np

class GraphBuffer:
    def __init__(self, args):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]

        self.graphs = np.empty(
            (self.episode_length + 1, self.n_rollout_threads, ), dtype=object
        )

        self.step = 0
    
    def insert(self, graph):
        self.graphs[self.step + 1] = copy.deepcopy(graph)
        self.step = (self.step + 1) % self.episode_length
    
    def feed_forward_generator_actor(self, actor_num_mini_batch=None, mini_batch_size=None):
        """Training data generator for actor that uses MLP network."""
        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.graphs.shape[0:2]
        episode_length -= 1
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* the number of steps ({episode_length}) = {n_rollout_threads * episode_length}"
                f" is required to be greater than or equal to the number of actor mini batches ({actor_num_mini_batch})."
            )
            mini_batch_size = batch_size // actor_num_mini_batch

        graphs = self.graphs[:-1].reshape(-1, )
        # shuffle indices
        # rand = torch.randperm(batch_size).numpy()
        rand = np.arange(batch_size)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        for indices in sampler:
            graph_batch = graphs[indices]
            yield graph_batch

    def after_update(self):
        self.graphs[0] = copy.deepcopy(self.graphs[-1])