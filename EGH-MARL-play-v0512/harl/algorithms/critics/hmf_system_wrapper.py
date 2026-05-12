"""A thin wrapper so HARL can save/restore HMF system parameters."""
import os
import torch

class HMFSystemWrapperCritic:
    def __init__(self, hmf_system, device):
        self.hmf = hmf_system
        self.device = device

    def turn_on_grad(self): 
        # HMF update manages grad on its own; keep no-op for runner compatibility.
        return
    def turn_off_grad(self):
        return

    def lr_decay(self, *args, **kwargs):
        return

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.hmf.state_dict(), os.path.join(save_dir, "hmf_system.pt"))

    def restore(self, model_dir):
        sd = torch.load(os.path.join(model_dir, "hmf_system.pt"), map_location=self.device)
        self.hmf.load_state_dict(sd)
