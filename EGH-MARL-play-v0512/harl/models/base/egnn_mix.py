# from torch import nn
# import torch
# import torch.nn.functional as F
# from harl.utils.models_tools import init, get_init_method
# from harl.models.base.model_util import GatedNonlinearity


# def aggregate(message, row_index, n_node, aggr='sum', mask=None):
#     """
#     The aggregation function (aggregate edge messages towards nodes)
#     :param message: The edge message with shape [M, K]
#     :param row_index: The row index of edges with shape [M]
#     :param n_node: The number of nodes, N
#     :param aggr: aggregation type, sum or mean
#     :param mask: the edge mask (used in mean aggregation for counting degree)
#     :return: The aggreagated node-wise information with shape [N, K]
#     """
#     result_shape = (n_node, message.shape[1])
#     result = message.new_full(result_shape, 0)  # [N, K]
#     row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, K]
#     result.scatter_add_(0, row_index, message)  # [N, K]
#     if aggr == 'sum':
#         pass
#     elif aggr == 'mean':
#         count = message.new_full(result_shape, 0)
#         ones = torch.ones_like(message)
#         if mask is not None:
#             ones = ones * mask.unsqueeze(-1)
#         count.scatter_add_(0, row_index, ones)
#         result = result / count.clamp(min=1)
#     else:
#         raise NotImplementedError('Unknown aggregation method:', aggr)
#     return result  # [N, K]


# class BaseMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
#         super(BaseMLP, self).__init__()
#         self.residual = residual
#         init_method = get_init_method('orthogonal_')
#         # active_func = get_active_func(activation_func)
#         gain = nn.init.calculate_gain('relu')
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
#         if flat:
#             activation = nn.Tanh()
#             hidden_dim = 4 * hidden_dim
#         if residual:
#             assert output_dim == input_dim
#         if last_act:
#             self.mlp = nn.Sequential(
#                 # init_(nn.Linear(input_dim, hidden_dim)),
#                 nn.Linear(input_dim, hidden_dim),
#                 activation,
#                 # nn.LayerNorm(hidden_dim),
#                 # init_(nn.Linear(hidden_dim, output_dim)),
#                 nn.Linear(hidden_dim, output_dim),
#                 activation
#             )
#         else:
#             self.mlp = nn.Sequential(
#                 # init_(nn.Linear(input_dim, hidden_dim)),
#                 nn.Linear(input_dim, hidden_dim),
#                 activation,
#                 # nn.LayerNorm(hidden_dim),    # add layer norm
#                 # init_(nn.Linear(hidden_dim, output_dim))
#                 nn.Linear(hidden_dim, output_dim)
#             )

#     def forward(self, x):
#         return self.mlp(x) if not self.residual else self.mlp(x) + x


# class InvariantScalarNet(nn.Module):
#     def __init__(self, n_vector_input, hidden_dim, output_dim, activation, n_scalar_input=0, norm=True, last_act=False,
#                  flat=False):
#         """
#         The universal O(n) invariant network using scalars.
#         :param n_vector_input: The total number of input vectors.
#         :param hidden_dim: The hidden dim of the network.
#         :param activation: The activation function.
#         """
#         super(InvariantScalarNet, self).__init__()
#         self.input_dim = n_vector_input * n_vector_input + n_scalar_input
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.activation = activation
#         self.norm = norm
#         self.scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.output_dim, self.activation, last_act=last_act,
#                                   flat=flat)

#     def forward(self, vectors, scalars=None):
#         """
#         :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
#         :param scalars: torch.Tensor with shape [N, L] (Optional)
#         :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]
#         """
#         if type(vectors) == list:
#             Z = torch.stack(vectors, dim=-1)  # [N, 2, V] V is the number of vectors
#         else:
#             Z = vectors
#         K = Z.shape[-1]
#         Z_T = Z.transpose(-1, -2)  # [N, V, 2]
#         scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, V, V]
#         scalar = scalar.reshape(-1, K * K)  # [N, VV]
#         if self.norm:
#             scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
#         if scalars is not None:
#             scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + L] L=257
#         scalar = self.scalar_net(scalar)  # [N, H]
#         return scalar


# class EGNN_Layer(nn.Module):
#     def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False, nonlinear=True):
#         super(EGNN_Layer, self).__init__()
#         self.with_v = with_v
#         self.edge_message_net = InvariantScalarNet(n_vector_input=1, hidden_dim=hidden_nf, output_dim=hidden_nf,
#                                                    activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
#                                                    norm=norm, last_act=True, flat=flat)
#         self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
#                                  flat=flat)
#         self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
#                                 activation=activation, flat=flat)
#         self.node_x_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation, 
#                                   flat=flat)
#         if self.with_v:
#             self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
#                                       flat=flat)
#         else:
#             self.node_v_net = None
#         self.nonlinear = nonlinear
#         self.gated1 = GatedNonlinearity(30, 2) 

#     def forward(self, x, h, edge_index, edge_fea, v=None, origin_h=None):
#         row, col = edge_index
#         rij = x[row] - x[col]  # [BM, 2], M=90 is the number of edges.
#         if edge_fea is None:
#             hij = torch.cat((h[row], h[col]), dim=-1)
#         else:
#             hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1], hij is the concat of hi, hj and edge feature
#         message = self.edge_message_net(vectors=[rij], scalars=hij)  # [B*90, H], message is mij
#         coord_message = self.coord_net(message)  # [B*90, 1]
        
#         f = (x[row] - x[col]) * coord_message  # [B*90, 2] update of mij
#         tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 2]
#         # use gated nolinear
#         # if self.nonlinear:
#         #     tot_f = self.gated1(origin_h, tot_f)
        
#         tot_f = torch.clamp(tot_f, min=-100, max=100) # tot_f is used to update x

#         if v is not None:
#             v = self.node_v_net(h) * v + tot_f
#             # x = self.node_x_net(h) * x + tot_f
#             x = x + v
#         else:
#             # x = x + tot_f  # [BN, 2]
#             x = self.node_x_net(h) * x + tot_f

#         tot_message = aggregate(message=message, row_index=row, n_node=x.shape[0], aggr='sum')  # [BN, H], sum of mij
#         node_message = torch.cat((h, tot_message), dim=-1)  # [BN, 2H]
#         h = self.node_net(node_message)  # [BN, H], update of h
#         return x, v, h


# class EGNN(nn.Module):
#     def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
#                  flat=False, norm=False):
#         super(EGNN, self).__init__()
#         self.layers = nn.ModuleList()
#         self.n_layers = n_layers
#         self.with_v = with_v
#         # input feature mapping
#         self.embedding = nn.Linear(in_node_nf, hidden_nf)
#         for i in range(self.n_layers):
#             nonlinear_flag = True if (i != self.n_layers-1) else False
#             layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm, nonlinear=nonlinear_flag)
#             self.layers.append(layer)
#         self.to(device)

#     def forward(self, x, h, edge_index, edge_fea, v=None):
#         origin_h = h
#         h = self.embedding(h)   # [BN, H]
#         for i in range(self.n_layers):
#             x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v, origin_h=origin_h) # x: [BN, 2], v: [BN, 2], h: [BN, H]
#         return (x, v, h) if v is not None else (x, h)



from torch import nn
import torch
import torch.nn.functional as F
from harl.utils.models_tools import init, get_init_method
from harl.models.base.model_util import GatedNonlinearity
import numpy as np

# 这个函数是聚合与某个节点相关的所有边信息
def aggregate(message, row_index, n_node, aggr='sum', mask=None):
    """
    The aggregation function (aggregate edge messages towards nodes)
    :param message: The edge message with shape [M, K]
    :param row_index: The row index of edges with shape [M]
    :param n_node: The number of nodes, N
    :param aggr: aggregation type, sum or mean
    :param mask: the edge mask (used in mean aggregation for counting degree)
    :return: The aggreagated node-wise information with shape [N, K]
    """
    result_shape = (n_node, message.shape[1])
    result = message.new_full(result_shape, 0)  # [N, K]
    row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, K]
    result.scatter_add_(0, row_index, message)  # [N, K]
    if aggr == 'sum':
        pass
    elif aggr == 'mean':
        count = message.new_full(result_shape, 0)
        ones = torch.ones_like(message)
        if mask is not None:
            ones = ones * mask.unsqueeze(-1)
        count.scatter_add_(0, row_index, ones)
        result = result / count.clamp(min=1)
    else:
        raise NotImplementedError('Unknown aggregation method:', aggr)
    return result  # [N, K]


# 多层感知机 包含两层全连接层
class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        init_method = get_init_method('orthogonal_')
        # active_func = get_active_func(activation_func)
        gain = nn.init.calculate_gain('relu')
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                # init_(nn.Linear(input_dim, hidden_dim)),
                nn.Linear(input_dim, hidden_dim),
                activation,
                # nn.LayerNorm(hidden_dim),
                # init_(nn.Linear(hidden_dim, output_dim)),
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                # init_(nn.Linear(input_dim, hidden_dim)),
                nn.Linear(input_dim, hidden_dim),
                activation,
                # nn.LayerNorm(hidden_dim),    # add layer norm
                # init_(nn.Linear(hidden_dim, output_dim))
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x
    
    

# 用于处理边与边的信息
class InvariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, output_dim, activation, n_scalar_input=0, norm=True, last_act=False,
                 flat=False):
        """
        The universal O(n) invariant network using scalars.
        :param n_vector_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(InvariantScalarNet, self).__init__()
        self.input_dim = n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.norm = norm
        self.scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.output_dim, self.activation, last_act=last_act,
                                  flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 2, V] V is the number of vectors
        else:
            Z = vectors
        # K = Z.shape[-1]
        # Z_T = Z.transpose(-1, -2)  # [N, V, 2]
        # scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, V, V]
        # scalar = scalar.reshape(-1, K * K)  # [N, VV]
        scalar = torch.norm(Z, p=2, dim=1)  # [N, V]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + L] L=257
        scalar = self.scalar_net(scalar)  # [N, H]
        return scalar


class EGNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False, nonlinear=True):
        super(EGNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = InvariantScalarNet(n_vector_input=2, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                                   activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                   norm=norm, last_act=True, flat=flat)
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)
        self.node_x_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation, 
                                  flat=flat)
        if self.with_v:
            self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                      flat=flat)
        else:
            self.node_v_net = None
        # self.nonlinear = nonlinear
        # self.gated1 = GatedNonlinearity(30, 2) 

    def forward(self, x, h, edge_index, edge_fea, v=None, origin_h=None):
        row, col = edge_index
        rij = x[row] - x[col]  # [BM, 2], M=90 is the number of edges.
        vij = v[row] - v[col]
        if edge_fea is None:
            hij = torch.cat((h[row], h[col]), dim=-1)
        else:
            hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1], hij is the concat of hi, hj and edge feature
        message = self.edge_message_net(vectors=[rij, vij], scalars=hij)  # [B*90, H], message is mij
        coord_message = self.coord_net(message)  # [B*90, 1]
        
        f = (x[row] - x[col]) * coord_message  # [B*90, 2] update of mij
        fv = (v[row] - v[col]) * coord_message
        tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 2]
        tot_fv = aggregate(message=fv, row_index=row, n_node=x.shape[0], aggr='mean')  # 使用这一项更新v效果不好
        # use gated nolinear
        # if self.nonlinear:
        #     tot_f = self.gated1(origin_h, tot_f)
        
        tot_f = torch.clamp(tot_f, min=-100, max=100) # tot_f is used to update x
        tot_fv = torch.clamp(tot_fv, min=-100, max=100)

        if v is not None:
            v = self.node_v_net(h) * v + tot_f # 对的
            x = self.node_x_net(h) * x + tot_f
            # x = x + v
        else:
            # x = x + tot_f  # [BN, 2]
            x = self.node_x_net(h) * x + tot_f

        tot_message = aggregate(message=message, row_index=row, n_node=x.shape[0], aggr='sum')  # [BN, H], sum of mij
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, 2H]
        h = self.node_net(node_message)  # [BN, H], update of h
        return x, v, h


class EGNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.with_v = with_v
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            nonlinear_flag = True if (i != self.n_layers-1) else False
            layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm, nonlinear=nonlinear_flag)
            self.layers.append(layer)
        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        origin_h = h
        h = self.embedding(h)   # [BN, H]
        for i in range(self.n_layers):
            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v, origin_h=origin_h) # x: [BN, 2], v: [BN, 2], h: [BN, H]
        return (x, v, h) if v is not None else (x, h)




class xyMLP(nn.Module):
    """
    一个可指定层数的多层感知机 (MLP)。
    
    输入形状: (n, h)  ->  n 为批量大小, h 为输入特征维度
    输出形状: (n, m)  ->  输出为二分类 logits 或回归值
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, l: int):
        """
        参数:
            input_dim (int): 输入特征的维度 h
            hidden_dim (int): 隐藏层的神经元数量（所有隐藏层使用相同的维度）
            output_dim (int): 输出维度
            l (int): 网络的层数（隐藏层的数量）。必须 >= 1。
                    当 l=1 时，网络结构为 [input_dim -> hidden_dim -> 2]。
                    当 l=2 时，网络结构为 [input_dim -> hidden_dim -> hidden_dim -> 2]。
                    以此类推。
        """
        super(xyMLP, self).__init__()
        assert l >= 1, "层数 l 必须至少为 1"
        
        # 使用 ModuleList 来存储所有线性层
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # 添加后续的隐藏层（如果有）
        for _ in range(l - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 输出层：从最后一个隐藏层映射到 2 维输出
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 可选：记录层数，以备后用
        self.num_hidden_layers = l
        
    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (n, input_dim)
        
        返回:
            torch.Tensor: 输出张量，形状为 (n, 2)
        """
        # 通过所有隐藏层，并在每层后应用 ReLU 激活函数
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # 通过输出层（不加激活函数，常用于分类 logits 或回归输出）
        out = self.output_layer(x)
        return out

