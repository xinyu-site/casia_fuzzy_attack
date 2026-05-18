"""Dummy critic for HMF.

OffPolicyBaseRunner requires a critic object. For HMF we do not use HARL critic training
loop; instead, OffPolicyHMFRunner performs updates via HMFAgentSystem directly.
This dummy critic is only to satisfy construction; it will be replaced in runner.__init__.
"""
class HMFDummyCritic:
    def __init__(self, *args, **kwargs):
        pass
    def turn_on_grad(self): pass
    def turn_off_grad(self): pass
    def lr_decay(self, *args, **kwargs): pass
    def save(self, *args, **kwargs): pass
    def restore(self, *args, **kwargs): pass
