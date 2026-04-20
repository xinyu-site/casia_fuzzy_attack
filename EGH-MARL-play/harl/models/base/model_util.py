from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import torch.nn.init as torch_init
from torch.nn.functional import cosine_similarity
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class edge_tool():
    def __init__(self, agent, net_type="egnn"):
        self.n_threads = agent.n_threads
        self.episode_length = agent.episode_length
        self.n_nodes = agent.n_nodes
        self.device = agent.device
        self.net_type = net_type
        self.actor_num_mini_batch = None
    
    def get_edges(self, adjacency_matrix=None):
        rows, cols = [], []
        if adjacency_matrix is None:
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
        else:
            n_nodes = adjacency_matrix.shape[0]
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adjacency_matrix[i, j] == 1 and i != j:
                        rows.append(i)
                        cols.append(j)
        edges = [rows, cols]
        return edges
    
    def get_edges_batch(self, batch_size, local_obs=None):
        if local_obs is None:
            edges = self.get_edges()
            edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
            if batch_size > 1:
                rows, cols = [], []
                for i in range(batch_size):
                    rows.append(edges[0] + self.n_nodes * i)
                    cols.append(edges[1] + self.n_nodes * i)
                edges = [torch.cat(rows), torch.cat(cols)]
            edges[0] = edges[0].to(self.device)
            edges[1] = edges[1].to(self.device)
            if not self.net_type == "egnn":
                edges = torch.stack(edges)
            return edges
        else:
            all_edges_row = []
            all_edges_col = []
            for batch_idx in range(batch_size):
                adjacency_matrix = local_obs[batch_idx, :, -self.n_nodes:].reshape((self.n_nodes, self.n_nodes))
                edges = self.get_edges(adjacency_matrix)
                all_edges_row.extend([e + batch_idx * self.n_nodes for e in edges[0]])
                all_edges_col.extend([e + batch_idx * self.n_nodes for e in edges[1]])

            all_edges = [torch.LongTensor(all_edges_row).to(self.device), torch.LongTensor(all_edges_col).to(self.device)]
            if not self.net_type == "egnn":
                all_edges = torch.stack(all_edges)
            return all_edges
    
    # # 专用于HEPN，构建Networkx图
    # def get_graphs(self, batch_size, equ_nf, local_obs=None):
    #     graphs = []
    #     if local_obs is None:
    #         edges = self.get_edges()
    #         edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    #         if batch_size > 1:
    #             rows, cols = [], []
    #             for i in range(batch_size):
    #                 G = nx.Graph()
    #                 G.add_nodes_from(range(self.n_nodes))
    #                 G.add_edges_from(torch.stack(edges).T.tolist())
    #                 graphs.append({'G': G})
    #                 rows.append(edges[0] + self.n_nodes * i)
    #                 cols.append(edges[1] + self.n_nodes * i)
    #             edges = [torch.cat(rows), torch.cat(cols)]
    #         edges[0] = edges[0].to(self.device)
    #         edges[1] = edges[1].to(self.device)
    #         return edges, graphs
    #     else:
    #         all_edges_row = []
    #         all_edges_col = []
    #         adj_list = []
    #         for batch_idx in range(batch_size):
    #             G = nx.Graph()
    #             adjacency_matrix = local_obs[batch_idx, :, -self.n_nodes:].reshape((self.n_nodes, self.n_nodes))
    #             adj_list.append(adjacency_matrix)
    #             edges = self.get_edges(adjacency_matrix)
                
    #             edge_weights = cosine_similarity(torch.tensor(local_obs[batch_idx, edges[0], :-(equ_nf + self.n_nodes)]), 
    #                                              torch.tensor(local_obs[batch_idx, edges[1], :-(equ_nf + self.n_nodes)]), dim=1)
    #             edge_weights = torch.exp(edge_weights)
    #             edge_weights = edge_weights.tolist()
    #             G.add_nodes_from(range(self.n_nodes))
    #             G.add_weighted_edges_from([[e[0], e[1], w] for e, w in zip(torch.tensor(edges).T.tolist(), edge_weights)])
    #             graphs.append({'G': G})

    #             all_edges_row.extend([e + batch_idx * self.n_nodes for e in edges[0]])
    #             all_edges_col.extend([e + batch_idx * self.n_nodes for e in edges[1]])

    #         all_edges = [torch.LongTensor(all_edges_row).to(self.device), torch.LongTensor(all_edges_col).to(self.device)]
            
    #         return all_edges, graphs

    # # 专用于HEPN，构建Networkx图
    # def get_graphs_critic(self, edges, equ_nf, cent_obs):
    #     graphs = []
    #     batch_size = cent_obs.shape[0]
    #     if batch_size > 1:
    #         for i in range(batch_size):
    #             G = nx.Graph()
    #             edge_weights = cosine_similarity(torch.tensor(cent_obs[i, edges[0], :-equ_nf]), 
    #                                              torch.tensor(cent_obs[i, edges[1], :-equ_nf]), dim=1)
    #             edge_weights = torch.exp(edge_weights)
    #             edge_weights = edge_weights.tolist()
    #             G.add_nodes_from(range(self.n_nodes))
    #             G.add_weighted_edges_from([[e[0], e[1], w] for e, w in zip(torch.tensor(edges).T.tolist(), edge_weights)])
    #             graphs.append({'G': G})
    #     return graphs


class LocalMLP(nn.Module):
    # 对局部信息进行处理
    def __init__(self, input_dim, output_dim):
        super(LocalMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = torch.mean(x, dim=1)
        return x
    
class HyperMLP(nn.Module):
    # 利用超网络对局部信息进行处理
    def __init__(self, input_dim, output_dim, hyper_hidden_dim):
        super(HyperMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.hyper_module = nn.Sequential(
            nn.Linear(self.input_dim ,self.hyper_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hyper_hidden_dim, self.output_dim * self.input_dim)      
        )
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        size1 = x.shape[1]
        Wx = self.hyper_module(x)
        Wx = Wx.reshape(-1, size1, self.input_dim, self.output_dim)
        x = torch.matmul(x.unsqueeze(2), Wx)
        x = torch.sum(x, dim=1).squeeze()
        x = self.fc(x)
        return x
    
class AttentionLayer(nn.Module):
    # 仅后equ_dim维采用时序加权
    def __init__(self, input_features, output_features, total_dim):
        super(AttentionLayer, self).__init__()
        self.windows_size = input_features
        self.equ_dim = output_features
        self.total_dim = total_dim
        self.inv_dim = total_dim - output_features
        self.weights = nn.Linear(self.windows_size*self.inv_dim, self.windows_size*self.equ_dim)

    def forward(self, x):
        h = x[:, :self.inv_dim, :]
        h = h.reshape(-1, self.windows_size*self.inv_dim)
        attention_scores = self.weights(h)
        attention_scores = attention_scores.reshape(-1, self.equ_dim, self.windows_size)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # attention_scores = torch.tanh(attention_scores)
        # attention_scores = torch.sigmoid(attention_scores)
        
        first_part = x[:, :self.inv_dim, -1]
        second_part = x[:, self.inv_dim:, :]
        second_part = second_part * attention_scores
        second_part = torch.sum(second_part, dim=-1)
        combined = torch.cat([first_part, second_part], dim=1)
        return combined
    
class PartLinearLayer(nn.Module):
    # 仅后equ_dim维采用时序加权
    def __init__(self, input_features, output_features, total_dim):
        super(PartLinearLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))
        torch_init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # 使用 He 初始化
        self.equ_dim = output_features
        self.total_dim = total_dim
        self.inv_dim = total_dim - output_features

    def forward(self, x):
        first_part = x[:, :self.inv_dim, -1]
        second_part = x[:, self.inv_dim:, :]
        second_part = second_part * self.weights
        second_part = torch.sum(second_part, dim=-1)
        combined = torch.cat([first_part, second_part], dim=1)
        return combined
        
class AllLinearLayer(nn.Module):
    # 全部维度都采用时序加权
    def __init__(self, input_features, output_features):
        super(AllLinearLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(output_features, input_features))
        torch_init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # 使用 He 初始化

    def forward(self, x):
        # 利用广播机制进行元素级乘法
        weighted = x * self.weights
        # 在最后一个维度上求和
        return torch.sum(weighted, dim=-1)

class StaticWeightLinearLayer(nn.Module):
    # 检查数据正确性，通过矩阵仅最后一行为1别的为0实现仅使用最新时刻的数据
    def __init__(self, input_features, output_features):
        super(StaticWeightLinearLayer, self).__init__()
        self.static_linear = nn.Linear(in_features=input_features, out_features=output_features, bias=False)
        weight = torch.zeros((output_features, input_features))
        weight[:, -1] = 1.0
        # 设置静态权重并禁用梯度
        self.static_linear.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = self.static_linear(x)
        x = torch.squeeze(x, dim=-1)
        return x
      
class GatedNonlinearity(nn.Module):
    def __init__(self, dim_h, dim_x):
        super(GatedNonlinearity, self).__init__()
        # 调整 h 到 x 的维度
        self.map_h = nn.Linear(dim_h, dim_x)
        self.gate = nn.Sigmoid()

    def forward(self, h, x):
        # 调整 h 的维度
        h_mapped = self.map_h(h)
        # 应用门控
        gated_output = x * self.gate(h_mapped)
        return gated_output

def rotation_obs3d(obs, angle, equ_nf_dim):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                [np.sin(angle), np.cos(angle)]])
    equ_feature = obs[:, :, -equ_nf_dim:]
    pos = equ_feature[:, :, :2]
    vel = equ_feature[:, :, 2:]
    rotated_pos = np.einsum('ijk,kl->ijl', pos, rotation_matrix.T)
    rotated_vel = np.einsum('ijk,kl->ijl', vel, rotation_matrix.T)
    rotated_features = np.concatenate((rotated_pos, rotated_vel), axis=-1)
    obs[:, :, -equ_nf_dim:] = rotated_features
    return obs

def rotation_obs2d(obs, angle, equ_nf_dim):
    equ_feature = obs[:, -equ_nf_dim:]
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                [np.sin(angle), np.cos(angle)]])
    pos = equ_feature[:, :2]
    vel = equ_feature[:, 2:]
    rotated_pos = np.dot(pos, rotation_matrix.T)
    rotated_vel = np.dot(vel, rotation_matrix.T)
    rotated_features = np.hstack((rotated_pos, rotated_vel))
    obs[:, -equ_nf_dim:] = rotated_features
    return obs

# def rotation_action(action, angle):
#     # 定义旋转矩阵
#     if isinstance(action, torch.Tensor):
#         action = action.detach().cpu().numpy()
#     rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
#                                 [np.sin(angle), np.cos(angle)]])
#     # 使用矩阵乘法对整个数组应用旋转
#     rotated_action = np.dot(action, rotation_matrix.T)
#     return rotated_action

def rotation_action(action, angle):
    angle = torch.tensor(angle)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation_matrix = torch.tensor([[cos_angle, -sin_angle], 
                                    [sin_angle, cos_angle]])
    rotation_matrix = rotation_matrix.to(device)
    rotated_action = torch.matmul(action, rotation_matrix.T)
    return rotated_action

