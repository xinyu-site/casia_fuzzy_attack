import torch

a = torch.rand(2)
rot_mat = torch.tensor([
    [0.0, -1.0], # [-1.0, 0.0], # [0.0, -1.0],
    [-1.0, 0.0] # [0.0, -1.0]  # [1.0, 0.0]
])
rot_a = torch.matmul(rot_mat, a)

dir_temp = torch.tensor([
    [0.0, 1.0],  # up
    [0.0, -1.0], # down
    [-1.0, 0.0], # left
    [1.0, 0.0]   # right
])
logits_a = torch.matmul(a, dir_temp.T)
logits_rot_a = torch.matmul(rot_a, dir_temp.T)

print("logits_a:{}".format(logits_a))
print("logits_rot_a:{}".format(logits_rot_a))