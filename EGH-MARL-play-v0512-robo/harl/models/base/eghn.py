from torch import nn
import torch
import torch.nn.functional as F
from torch_sparse import spmm
from torch_sparse import spmm
from harl.utils.models_tools import init, get_active_func, get_init_method
import scipy.sparse as sp
# from harl.models.base.gnn_basic import EGNN, EquivariantScalarNet, BaseMLP, aggregate, EGMN


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


class EquivariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=True):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input, self.activation, flat=flat)
        self.out_scalar_net = BaseMLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.activation, flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 2, V]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, V, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, V, V]
        scalar = scalar.reshape(-1, K * K)  # [N, VV]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + 256]
        scalar = self.in_scalar_net(scalar)  # [N, H]
        vec_scalar = self.out_vector_net(scalar)  # [N, V]
        vector = torch.einsum('bij,bj->bi', Z, vec_scalar)  # [N, 2]
        scalar = self.out_scalar_net(scalar)  # [N, H]

        return vector, scalar


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
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
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
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, V, 2]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, V, V]
        scalar = scalar.reshape(-1, K * K)  # [N, VV]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + L] L=257
        scalar = self.scalar_net(scalar)  # [N, H]
        return scalar


class EGNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False):
        super(EGNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = InvariantScalarNet(n_vector_input=1, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                                   activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                   norm=norm, last_act=True, flat=flat)
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)
        if self.with_v:
            self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                      flat=flat)
        else:
            self.node_v_net = None

    def forward(self, x, h, edge_index, edge_fea, v=None):
        row, col = edge_index
        rij = x[row] - x[col]  # [BM, 2], M=90 is the number of edges.
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1], hij is the concat of hi, hj and edge feature
        message = self.edge_message_net(vectors=[rij], scalars=hij)  # [B*90, H], message is mij
        coord_message = self.coord_net(message)  # [B*90, 1]
        f = (x[row] - x[col]) * coord_message  # [B*90, 2] update of mij
        tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 2]
        tot_f = torch.clamp(tot_f, min=-100, max=100) # tot_f is used to update x

        if v is not None:
            v = self.node_v_net(h) * v + tot_f
            x = x + v
        else:
            x = x + tot_f  # [BN, 2]

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
            layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            self.layers.append(layer)
        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        h = self.embedding(h)   # [BN, H]
        for i in range(self.n_layers):
            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v) # x: [BN, 2], v: [BN, 2], h: [BN, H]
        return (x, v, h) if v is not None else (x, h)


class EGMN(nn.Module):
    def __init__(self, n_layers, n_vector_input, hidden_dim, n_scalar_input, activation=nn.SiLU(), norm=False, flat=False):
        super(EGMN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            cur_layer = EquivariantScalarNet(n_vector_input=n_vector_input + i, hidden_dim=hidden_dim,
                                             activation=activation, n_scalar_input=n_scalar_input if i == 0 else hidden_dim,
                                             norm=norm, flat=flat)
            self.layers.append(cur_layer)

    def forward(self, vectors, scalars):
        cur_vectors = vectors
        for i in range(self.n_layers):
            vector, scalars = self.layers[i](cur_vectors, scalars)
            cur_vectors.append(vector)
        return cur_vectors[-1], scalars


class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1))
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time


class EquivariantEdgeScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=False):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantEdgeScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input * n_vector_input,
                                      self.activation, flat=flat)

    def forward(self, vectors_i, vectors_j, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        Z_i, Z_j = vectors_i, vectors_j  # [N, 2, V]
        K = Z_i.shape[-1]
        Z_j_T = Z_j.transpose(-1, -2)  # [N, V, 2]
        scalar = torch.einsum('bij,bjk->bik', Z_j_T, Z_i)  # [N, V, V]
        scalar = scalar.reshape(-1, K * K)  # [N, VV]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + 257]
        scalar = self.in_scalar_net(scalar)  # [N, H], hij
        vec_scalar = self.out_vector_net(scalar)  # [N, VV]
        vec_scalar = vec_scalar.reshape(-1, Z_j.shape[-1], Z_i.shape[-1])  # [N, V, V], Hij
        vector = torch.einsum('bij,bjk->bik', Z_j, vec_scalar)  # [N, 2, V], Mij
        return vector, scalar


class PoolingLayer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, n_vector_input, activation=nn.SiLU(), flat=False):
        super(PoolingLayer, self).__init__()
        self.edge_message_net = EquivariantEdgeScalarNet(n_vector_input=n_vector_input, hidden_dim=hidden_nf,
                                                         activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                         norm=True, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)

    def forward(self, vectors, h, edge_index, edge_fea):
        """
        :param vectors: the node vectors with shape: [BN, 3, V] where V is the number of vectors
        :param h: the scalar node feature with shape: [BN, K]
        :param edge_index: the edge index with shape [2, BM]
        :param edge_fea: the edge feature with shape: [BM, T]
        :return: the updated node vectors [BN, 3, V] and node scalar feature [BN, K]
        """
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1]
        vectors_i, vectors_j = vectors[row], vectors[col]  # [B*90, 3, V]
        vectors_out, message = self.edge_message_net(vectors_i=vectors_i, vectors_j=vectors_j, scalars=hij)  # [B*90, 3, V], vector_out is Mij, message is hij
        DIM, V = vectors_out.shape[-2], vectors_out.shape[-1]
        vectors_out = vectors_out.reshape(-1, DIM * V)  # [B*90, 2V]
        vectors_out = aggregate(message=vectors_out, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, 2V]
        vectors_out = vectors_out.reshape(-1, DIM, V)  # [BN, 2, V]
        vectors_out = vectors + vectors_out  # [BN, 2, V], update of Z
        tot_message = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='sum')  # [BN, H]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, 2H]
        h = self.node_net(node_message) + h  # [BN, H], update of h  与论文中的更新公式不匹配，原文中通过mlp后直接作为h的更新，这里还额外做了加法
        return vectors_out, h


class PoolingNet(nn.Module):
    def __init__(self, n_layers, in_edge_nf, n_vector_input,
                 hidden_nf, output_nf, activation=nn.SiLU(), device='cpu', flat=False):
        super(PoolingNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            layer = PoolingLayer(in_edge_nf, hidden_nf, n_vector_input=n_vector_input, activation=activation, flat=flat)
            self.layers.append(layer)
        self.pooling = nn.Sequential(
            nn.Linear(hidden_nf, 8 * hidden_nf),
            nn.Tanh(),
            nn.Linear(8 * hidden_nf, output_nf)
        )
        self.to(device)

    def forward(self, vectors, h, edge_index, edge_fea):
        """
        :param vectors: [x - x_mean, newx - x, v]
        :edge_index: local_edge_index, connectivity-based edges
        """
        if type(vectors) == list:
            vectors = torch.stack(vectors, dim=-1)  # [BN, 2, V]
        for i in range(self.n_layers):
            vectors, h = self.layers[i](vectors, h, edge_index, edge_fea)  # vectors: [BN, 2, V], h: [BN, H]
        pooling = self.pooling(h)  # [BN, C]  C is the number of clusters
        return pooling, vectors, h  # [BN, C]  此处没有返回vectors，不知是何原因，pooling将用于softmax计算聚类分数


class EGHN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, n_cluster, layer_per_block=3, layer_pooling=3, layer_decoder=1,
                 flat=False, activation=nn.SiLU(), device='cpu', norm=False):
        super(EGHN, self).__init__()
        node_hidden_dim = hidden_nf
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.current_pooling_plan = None
        self.n_cluster = n_cluster  # 4 for simulation and 5 for mocap
        self.n_layer_per_block = layer_per_block
        self.n_layer_pooling = layer_pooling
        self.n_layer_decoder = layer_decoder
        self.flat = flat
        # low-level force net
        self.low_force_net = EGNN(n_layers=self.n_layer_per_block,
                                  in_node_nf=hidden_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf,
                                  activation=activation, device=device, with_v=True, flat=flat, norm=norm)
        self.low_pooling = PoolingNet(n_vector_input=4, hidden_nf=hidden_nf, output_nf=self.n_cluster,
                                      activation=activation, in_edge_nf=in_edge_nf, n_layers=self.n_layer_pooling, flat=flat)
        self.high_force_net = EGNN(n_layers=self.n_layer_per_block,
                                   in_node_nf=hidden_nf, in_edge_nf=1, hidden_nf=hidden_nf,
                                   activation=activation, device=device, with_v=True, flat=flat)
        if self.n_layer_decoder == 1:
            self.kinematics_net = EquivariantScalarNet(n_vector_input=3,
                                                       hidden_dim=hidden_nf,
                                                       activation=activation,
                                                       n_scalar_input=node_hidden_dim + node_hidden_dim,
                                                       norm=True,
                                                       flat=flat)
        else:
            self.kinematics_net = EGMN(n_vector_input=4, hidden_dim=hidden_nf, activation=activation,
                                       n_scalar_input=node_hidden_dim + node_hidden_dim, norm=True, flat=flat,
                                       n_layers=self.n_layer_decoder)

        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, local_edge_index, local_edge_fea, n_node, flag, v=None, node_mask=None, node_nums=None):
        """
        :param x: input positions [B * N, 3]
        :param h: input node feature [B * N, R]
        :param edge_index: edge index of the graph [2, B * M]
        :param edge_fea: input edge feature [B* M, T]
        :param local_edge_index: the edges used in pooling network [B * M']
        :param local_edge_fea: the feature of local edges [B * M', T]
        :param n_node: number of nodes per graph [1, ]
        :param v: input velocities [B * N, 3] (Optional)
        :param node_mask: the node mask when number of nodes are different in graphs [B * N, ] (Optional)
        :param node_nums: the real number of nodes in each graph
        :return:
        """
        h = self.embedding(h)  # [BN, 36] -> [BN, H]
        row, col = edge_index  # 全局观测下维度[B * 90]，完全图；局部观测下根据智能体距离构建邻接矩阵

        ''' low level force '''
        if v is not None:
            new_x, new_v, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)  # new_x: [BN, 2], new_v: [BN, 2], h: [BN, H]
        else:
            new_x, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)
        nf = new_x - x  # [BN, 2]
        nf_v = new_v - v

        ''' pooling network '''
        if node_nums is None:
            x_mean = torch.mean(x.reshape(-1, n_node, x.shape[-1]), dim=1, keepdim=True).expand(-1, n_node, -1).reshape(-1, x.shape[-1])  # [BN, 2]
        else:
            pooled_mean = (torch.sum(x.reshape(-1, n_node, x.shape[-1]), dim=1).T/node_nums).T.unsqueeze(dim=1) #[B,1,3]
            x_mean = pooled_mean.expand(-1, n_node, -1).reshape(-1, x.shape[-1])

        pooling_fea, vctors, h_ = self.low_pooling(vectors=[x - x_mean, nf, v, nf_v], h=h,
                                       edge_index=local_edge_index, edge_fea=local_edge_fea)  # [BN, C] C is the number of clusters
        '''hard_pooling 用于显示聚类结果'''
        # hard_pooling = self._force_k_hard_pooling(pooling_fea, n_node, node_mask=node_mask)
        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)
        self.current_pooling_plan = hard_pooling  # record the pooling plan
        # self.inspect_pooling_plan()
        if flag:
            # self.structural_entropy = self.get_structural_entropy(hard_pooling, h_.reshape(-1, n_node, h_.shape[-1]))
            self.structural_entropy = self.get_structural_entropy(edge_index, hard_pooling, h_)

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, C]

        sT = s.transpose(-2, -1)  # [B, C, N]
        p_index = torch.ones_like(nf)[..., 0]  # [BN, ]
        if node_mask is not None:
            p_index = p_index * node_mask
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, C, 1]  对st的行求和
        _x, _h, _nf = x.reshape(-1, n_node, x.shape[-1]), h.reshape(-1, n_node, h.shape[-1]), nf.reshape(-1, n_node, nf.shape[-1])
        _v = v.reshape(-1, n_node, v.shape[-1])
        _nf_v = nf_v.reshape(-1, n_node, nf_v.shape[-1])
        # _x: [B, N, 2], _h: [B, N, H], _nf: [B, N, 2], v: [B, N, 2]

        X, H, NF = torch.einsum('bij,bjk->bik', sT, _x), torch.einsum('bij,bjk->bik', sT, _h), torch.einsum('bij,bjk->bik', sT, _nf)
        V = torch.einsum('bij,bjk->bik', sT, _v)
        NF_V = torch.einsum('bij,bjk->bik', sT, _nf_v)
        X, H, NF, V, NF_V = X / count, H / count, NF / count, V / count, NF_V / count  
        # X: [B, C, 2], H: [B, C, H], NF: [B, C, 2], V: [B, C, 2]
        X, H, NF = X.reshape(-1, X.shape[-1]), H.reshape(-1, H.shape[-1]), NF.reshape(-1, NF.shape[-1])  # X, NF: [BC, 2], H: [BC, H]
        V = V.reshape(-1, V.shape[-1])  # [BC, 2]
        NF_V = NF_V.reshape(-1, NF_V.shape[-1])
        a = spmm(torch.stack((local_edge_index[0], local_edge_index[1]), dim=0),
                 torch.ones_like(local_edge_index[0]), x.shape[0], x.shape[0], pooling)  # [BN, C]
        a = a.reshape(-1, n_node, a.shape[-1])  # [B, N, C]
        '''
        A is constructed by connectivity-based local edges
        AA is constructed by distance-based global edges
        '''
        A = torch.einsum('bij,bjk->bik', sT, a)  # [B, C, C]
        self.cut_loss = self.get_cut_loss(A)
        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), x.shape[0], x.shape[0], pooling)  # [BN, C]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, C]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, C, C]

        # construct high-level edges
        h_row, h_col, h_edge_fea, h_edge_mask = self.construct_edges(AA, AA.shape[-1])  # [BCC]
        ''' high-level message passing '''
        self.h_edge_index = torch.stack((h_row, h_col), dim=0)
        h_new_x, h_new_v, h_new_h = self.high_force_net(X, H, (h_row, h_col), h_edge_fea.unsqueeze(-1), v=V)  # h_new_x, h_new_v: [BC, 2], h_new_h: [BC, H]
        h_nf = h_new_x - X
        h_nf_v = h_new_v - V

        ''' high-level kinematics update '''
        # _X = X + h_nf  # [BC, 2]
        _X = h_new_x  # [BC, 2]
        _V = h_new_v  # [BC, 2]
        _H = h_new_h  # [BC, H]

        ''' low-level kinematics update '''
        l_nf = h_nf.reshape(-1, AA.shape[1], h_nf.shape[-1])  # [B, C, 2]
        l_nf = torch.einsum('bij,bjk->bik', s, l_nf).reshape(-1, l_nf.shape[-1])  # [BN, 2]

        l_nf_v = h_nf_v.reshape(-1, AA.shape[1], h_nf_v.shape[-1])
        l_nf_v = torch.einsum('bij,bjk->bik', s, l_nf_v).reshape(-1, l_nf_v.shape[-1])

        l_X = X.reshape(-1, AA.shape[1], X.shape[-1])  # [B, C, 2]
        l_X = torch.einsum('bij,bjk->bik', s, l_X).reshape(-1, l_X.shape[-1])  # [BN, 2]

        l_V = V.reshape(-1, AA.shape[1], V.shape[-1])  # [B, C, 2]
        l_V = torch.einsum('bij,bjk->bik', s, l_V).reshape(-1, l_V.shape[-1])  # [BN, 2]

        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, C, H]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, H]
        # l_kinematics, h_out = self.kinematics_net(vectors=[l_nf, x - l_X, v - l_V, nf],
        #                                           scalars=torch.cat((h, l_H), dim=-1))  # [BN, 3]
        l_kinematics, h_out = self.kinematics_net(vectors=[l_nf_v, v - l_V, nf_v], scalars=torch.cat((h, l_H), dim=-1))  # [BN, 2], [BN, H]
        _l_X = _X.reshape(-1, AA.shape[1], _X.shape[-1])  # [B, C, 2]
        _l_X = torch.einsum('bij,bjk->bik', s, _l_X).reshape(-1, _l_X.shape[-1])  # [BN, 2], Zagg

        _l_V = _V.reshape(-1, AA.shape[1], _V.shape[-1])
        _l_V = torch.einsum('bij,bjk->bik', s, _l_V).reshape(-1, _l_V.shape[-1])
        x_out = _l_X + l_kinematics  # [BN, 2]
        v_out = _l_V + l_kinematics - v

        return (x_out, v_out, h_out) if v is not None else (x_out, h_out)

    def inspect_pooling_plan(self):
        plan = self.current_pooling_plan  # [BN, C]
        if plan is None:
            print('No pooling plan!')
            return
        dist = torch.sum(plan, dim=0)  # [C,]
        # print(dist)
        dist = F.normalize(dist, p=1, dim=0)  # [C,]
        print('Pooling plan:', dist.detach().cpu().numpy())
        return

    def _force_k_hard_pooling(self, pooling_fea, n_node, node_mask=None):
        """
        Ensure each graph has at least one node assigned to each cluster.
        :param pooling_fea: [BN, C]
        :param n_node: number of nodes per graph
        :param node_mask: [BN,] optional, 1 for valid nodes
        :return: hard assignment indices [BN]
        """
        hard_idx = pooling_fea.argmax(dim=-1)  # [BN]
        if n_node <= 0:
            return hard_idx

        B = pooling_fea.shape[0] // n_node
        if B <= 0:
            return hard_idx

        scores = pooling_fea.reshape(B, n_node, self.n_cluster)
        hard_idx = hard_idx.reshape(B, n_node)
        if node_mask is not None:
            valid_mask = node_mask.reshape(B, n_node).bool()
        else:
            valid_mask = torch.ones((B, n_node), dtype=torch.bool, device=pooling_fea.device)

        for b in range(B):
            valid = valid_mask[b]
            if valid.sum() == 0:
                continue
            unique = torch.unique(hard_idx[b][valid])
            if unique.numel() >= self.n_cluster:
                continue

            assigned = ~valid.clone()
            hard_idx_b = hard_idx[b].clone()
            # pick one best node for each cluster among unassigned
            for c in range(self.n_cluster):
                if assigned.all():
                    break
                scores_c = scores[b, :, c].clone()
                scores_c[assigned] = -1e9
                idx = torch.argmax(scores_c)
                hard_idx_b[idx] = c
                assigned[idx] = True

            remaining = ~assigned
            if remaining.any():
                hard_idx_b[remaining] = scores[b, remaining].argmax(dim=-1)
            hard_idx[b] = hard_idx_b

        return hard_idx.reshape(-1)

    def get_cut_loss(self, A):
        A = F.normalize(A, p=2, dim=2)
        return torch.norm(A - torch.eye(A.shape[-1]).to(A.device), p="fro", dim=[1, 2]).mean()

    # def get_structural_entropy(self, hard_pooling, h):
    #     B, N, H = h.shape
    #     hard_pooling = hard_pooling.reshape(-1, N, self.n_cluster)
    #     norm_h = h / h.norm(dim=2, keepdim=True)
    #     cos_sim = torch.bmm(norm_h, norm_h.transpose(1, 2)) + 1
    #     mask = 1 - torch.eye(N, dtype=cos_sim.dtype, device=cos_sim.device).unsqueeze(0).expand(B, N, N)
    #     cos_sim = cos_sim * mask

    #     vol_G = cos_sim.sum(dim=(1, 2))
    #     degree = cos_sim.sum(dim=-1)

    #     nonzero_indices = torch.nonzero(hard_pooling, as_tuple=False)
    #     group = [[] for _ in range(B * self.n_cluster)]
    #     for idx in nonzero_indices:
    #         b, n, c = idx.tolist()
    #         # 计算每个B中每列的索引
    #         column_index = b * self.n_cluster + c
    #         group[column_index].append(n)
    #     # 重整索引列表，使其按B和C组织
    #     group = [group[i:i + self.n_cluster] for i in range(0, len(group), self.n_cluster)]

    #     batch_entropy = []
    #     for b in range(B):
    #         entropy = 0
    #         for j in range(self.n_cluster):
    #             nodes = group[b][j]
    #             if nodes:
    #                 V_j = degree[b][nodes].sum()
    #                 p = degree[b][nodes] / V_j
    #                 e1 = V_j / vol_G[b] * (p * torch.log2(p)).sum()
    #                 mask = torch.ones(10, dtype=torch.bool)
    #                 mask[nodes] = False
    #                 selected_rows = cos_sim[b, nodes, :]
    #                 selected_weights = selected_rows[:, mask]
    #                 g_j = selected_weights.sum()
    #                 e2 = g_j / vol_G[b] * torch.log2(V_j / vol_G[b])
    #                 entropy += (e1 + e2)
    #         batch_entropy.append(-entropy)

    #     e = sum(batch_entropy) / len(batch_entropy)
    #     return e

    def get_structural_entropy(self, edge_index, hard_pooling, h):
        start_indices, end_indices = edge_index[0], edge_index[1]
        edge_weights = torch.nn.functional.cosine_similarity(h[start_indices], h[end_indices], dim=1) + 1

        # Compute node partition
        nonzero_indices = torch.nonzero(hard_pooling, as_tuple=False)
        clusters_indices = nonzero_indices[:, 1]
        clusters = [nonzero_indices[clusters_indices == i, 0].tolist() for i in range(self.n_cluster)]

        vol_G = edge_weights.sum()
        total_loss = 0.0

        for c in range(self.n_cluster):
            cluster = clusters[c]
            if cluster:
                cluster_indices = torch.tensor(cluster, device=h.device)
                cluster_mask = (torch.isin(start_indices, cluster_indices) & torch.isin(end_indices, cluster_indices))
                cluster_edges_weights = edge_weights[cluster_mask]
                vol_C = cluster_edges_weights.sum()

                g_l = edge_weights[(torch.isin(start_indices, cluster_indices) ^ torch.isin(end_indices, cluster_indices))].sum()

                if cluster_edges_weights.shape[0] == 0:
                    cluster_loss = - (g_l / vol_G) * torch.log2((vol_C + g_l) / vol_G)
                else:
                    cluster_loss = -cluster_edges_weights.mean() - (g_l / vol_G) * torch.log2((vol_C + g_l) / vol_G)
                
                total_loss += cluster_loss

        return total_loss

    # def get_structural_entropy2(self, edge_index, hard_pooling, h):
    #     row, col = edge_index
    #     # 提取边的起始和终点节点的特征向量
    #     start_features = h[row]
    #     end_features = h[col]
    #     # 计算余弦相似度作为边权
    #     edge_weights = F.cosine_similarity(start_features, end_features) + 1
    #     # 获取节点总数
    #     num_nodes = h.size(0)
    #     # 初始化度向量
    #     degrees = torch.zeros(num_nodes, device=h.device)
    #     # 累加起始节点的边权重
    #     degrees.scatter_add_(0, row, edge_weights)
    #     # 累加终止节点的边权重
    #     degrees.scatter_add_(0, col, edge_weights)

    #     nonzero_indices = torch.nonzero(hard_pooling, as_tuple=False)
    #     clusters_indices = nonzero_indices[:, 1]
    #     clusters = [nonzero_indices[clusters_indices == i, 0].tolist() for i in range(self.n_cluster)]

    #     vol_G = degrees.sum()

    #     entropy = 0
    #     for j in range(self.n_cluster):
    #         nodes = clusters[j]
    #         if nodes:
    #             V_j = degrees[nodes].sum()
    #             p = degrees[nodes] / V_j
    #             e1 = V_j / vol_G * (p * torch.log2(p)).sum()

    #             mask = torch.zeros(max(row.max(), col.max()) + 1, device=h.device, dtype=torch.bool)
    #             mask[nodes] = True
    #             # 检查每条边是否连接了一个选定节点和一个非选定节点
    #             edges_connecting_outside = mask[row] != mask[col]
    #             # 计算这些边的权重和
    #             g_j = edge_weights[edges_connecting_outside].sum()
    #             # g_j = selected_weights.sum()
    #             e2 = g_j / vol_G * torch.log2(V_j / vol_G)
    #             entropy += (e1 + e2)
    #     return -entropy

    @staticmethod
    def construct_edges(A, n_node):
        h_edge_fea = A.reshape(-1)  # [BCC]
        h_row = torch.arange(A.shape[1]).unsqueeze(-1).expand(-1, A.shape[1]).reshape(-1).to(A.device)
        h_col = torch.arange(A.shape[1]).unsqueeze(0).expand(A.shape[1], -1).reshape(-1).to(A.device)
        h_row = h_row.unsqueeze(0).expand(A.shape[0], -1)
        h_col = h_col.unsqueeze(0).expand(A.shape[0], -1)
        offset = (torch.arange(A.shape[0]) * n_node).unsqueeze(-1).to(A.device)
        h_row, h_col = (h_row + offset).reshape(-1), (h_col + offset).reshape(-1)  # [BCC]
        h_edge_mask = torch.ones_like(h_row)  # [BCC]
        h_edge_mask[torch.arange(A.shape[1]) * (A.shape[1] + 1)] = 0
        return h_row, h_col, h_edge_fea, h_edge_mask


