from torch import nn
import torch

class MyMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 应用全连接层
        x = self.fc(x)
        # 计算每行的和，用于检测全0行
        row_sums = x.sum(dim=-1)
        # 创建一个mask，表示非全0的行
        non_zero_row_mask = row_sums != 0
        # 计算每个非全0行的个数，避免除以0
        non_zero_row_counts = non_zero_row_mask.sum(dim=1, keepdim=True)
        # 使用where条件操作来避免在全0行上的除法操作
        mean_values = torch.where(
            non_zero_row_mask.unsqueeze(2),
            x,
            torch.zeros_like(x)
        ).sum(dim=1) / non_zero_row_counts
        # 处理可能出现的除以0的情况（如果一个Batch全是0行）
        mean_values[torch.isnan(mean_values)] = 0
        return mean_values