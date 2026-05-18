from harl.utils.models_tools import init, get_init_method
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.typing import Adj, Size
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import spmm
from torch_sparse import SparseTensor, matmul
import numpy as np
import copy


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
    def __init__(
            self, 
            input_dim, hidden_dim, output_dim, 
            activation, residual=False, last_act=False, 
            flat=False
        ):
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
        if edge_fea is not None:
            hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1], hij is the concat of hi, hj and edge feature
        else:
            hij = torch.cat((h[row], h[col]), dim=-1)
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


class PoolingNet_v2(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)
    
    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class HEPN(nn.Module):
    def __init__(
            self, 
            in_node_nf, 
            in_edge_nf, 
            hidden_nf, 
            layer_per_block=3, 
            layer_pooling=3, 
            layer_decoder=1,
            flat=False, 
            activation=nn.SiLU(), 
            device='cpu', 
            norm=False
        ):
        super(HEPN, self).__init__()
        node_hidden_dim = hidden_nf
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.n_layer_per_block = layer_per_block
        # self.n_layer_pooling = layer_pooling
        self.n_layer_decoder = layer_decoder
        self.flat = flat
        self.device = device
        # low-level force net
        self.low_force_net = EGNN(
            n_layers=self.n_layer_per_block,
            in_node_nf=hidden_nf, 
            in_edge_nf=in_edge_nf, 
            hidden_nf=hidden_nf,
            activation=activation, 
            device=device, 
            with_v=True, 
            flat=flat, 
            norm=norm
        )
        self.pooling = PoolingNet_v2()
        self.high_force_net = EGNN(
            n_layers=self.n_layer_per_block,
            in_node_nf=hidden_nf, 
            in_edge_nf=0, 
            hidden_nf=hidden_nf,
            activation=activation, 
            device=device, 
            with_v=True, 
            flat=flat
        )
        if self.n_layer_decoder == 1:
            self.kinematics_net = EquivariantScalarNet(
                n_vector_input=3,
                hidden_dim=hidden_nf,
                activation=activation,
                n_scalar_input=node_hidden_dim + node_hidden_dim,
                norm=True,
                flat=flat
            )
        else:
            self.kinematics_net = EGMN(
                n_vector_input=4, 
                hidden_dim=hidden_nf, 
                activation=activation,
                n_scalar_input=node_hidden_dim + node_hidden_dim, 
                norm=True, 
                flat=flat,
                n_layers=self.n_layer_decoder
            )

        self.to(device)

    def __process_sep_edgeIndex(self, layer_data, layer=1):
        edge_mat_list = []
        start_pdx = [0]
        start_idx = [0]
        for i, data in enumerate(layer_data):
            mat = copy.deepcopy(data)
            start_pdx.append(start_pdx[i] + mat['node_size'][layer - 1])
            start_idx.append(start_idx[i] + mat['node_size'][layer])
            mat['interLayer_edgeMat'][layer][0, :] += start_idx[i]
            mat['interLayer_edgeMat'][layer][1, :] += start_pdx[i]
            edge_mat_list.append(mat['interLayer_edgeMat'][layer])
        edge_mat_list = torch.cat(edge_mat_list, 1)
        return edge_mat_list.to(self.device)

    def __process_sep_size(self, layer_data, layer=1):
        size = [(graph['node_size'][layer-1], graph['node_size'][layer]) for graph in layer_data]
        return np.array(size).sum(axis=0)
    
    def __process_layer_edgeIndex(self, layer_data, layer=0):
        edge_mat_list = []
        start_idx = [0]
        for i, data in enumerate(layer_data):
            mat = copy.deepcopy(data)
            start_idx.append(start_idx[i] + mat['node_size'][layer])
            mat['layer_edgeMat'][layer][0, :] += start_idx[i]
            mat['layer_edgeMat'][layer][1, :] += start_idx[i]
            edge_mat_list.append(mat['layer_edgeMat'][layer])
        edge_mat_list = torch.cat(edge_mat_list, 1)
        return edge_mat_list.to(self.device)

    def forward(self, x, h, edge_index, edge_fea, n_node, layer_data, v=None):
        """
        B: batch size
        N: 智能体数量
        n: 空间维度
        M: 等变特征数量
        :param equ_fea: 等变特征矩阵 [B * N, n, M]
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
        # n, M = equ.shape[-2], equ.shape[-1]
        h = self.embedding(h)  # [BN, 36] -> [BN, H]

        ''' low level force '''
        if v is not None:
            new_x, new_v, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)
        else:
            new_x, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)
        nf = new_x - x
        nf_v = new_v - v

        ''' derive high-level information (be careful with graph mini-batch) '''
        edge = self.__process_sep_edgeIndex(layer_data, 1)
        size = self.__process_sep_size(layer_data, 1)
        H = self.pooling(x=h, edge_index=edge, size=size)
        X = self.pooling(x=x, edge_index=edge, size=size)
        NF = self.pooling(x=nf, edge_index=edge, size=size)
        V = self.pooling(x=v, edge_index=edge, size=size)
        NF_V = self.pooling(x=nf_v, edge_index=edge, size=size)

        '''二维等变性测试代码，仅用于测试'''
        # theta = np.pi / 2
        # rotation_matrix = torch.tensor([
        #     [np.cos(theta), -np.sin(theta)], 
        #     [np.sin(theta), np.cos(theta)]
        # ], dtype=torch.float32)
        # rotation_matrices = rotation_matrix.expand(equ_fea.shape[0], -1, -1).to(self.device)
        # equ_fea_r = torch.bmm(rotation_matrices, equ_fea)
        # EQU_r = self.low_pooling(x=equ_fea_r.reshape(equ_fea_r.shape[0], -1), edge_index=edge, size=size)
        # rotation_matrices = rotation_matrix.expand(EQU.shape[0], -1, -1).to(self.device)
        # print(torch.bmm(rotation_matrices, EQU.reshape(-1, n, M)) == EQU_r.reshape(-1, n, M))

        '''构建高层图edge_index'''
        h_edge_index = self.__process_layer_edgeIndex(layer_data, 1)
        
        ''' high-level message passing '''
        h_new_x, h_new_v, h_new_h = self.high_force_net(X, H, h_edge_index, None, v=V)
        h_nf = h_new_x - X
        h_nf_v = h_new_v - V

        ''' high-level kinematics update '''
        _X = h_new_x  # [BC, 2]
        _V = h_new_v  # [BC, 2]
        _H = h_new_h  # [BC, H]

        ''' low-level kinematics update '''
        # size.reverse()
        l_nf = self.pooling(x=h_nf, edge_index=edge[[1, 0]], size=size[[1, 0]])
        l_nf_v = self.pooling(x=h_nf_v, edge_index=edge[[1, 0]], size=size[[1, 0]])
        l_X = self.pooling(x=X, edge_index=edge[[1, 0]], size=size[[1, 0]])
        l_V = self.pooling(x=V, edge_index=edge[[1, 0]], size=size[[1, 0]])
        l_H = self.pooling(x=_H, edge_index=edge[[1, 0]], size=size[[1, 0]])

        l_kinematics, h_out = self.kinematics_net(vectors=[l_nf_v, v - l_V, nf_v], scalars=torch.cat((h, l_H), dim=-1))

        _l_X = self.pooling(x=_X, edge_index=edge[[1, 0]], size=size[[1, 0]])
        _l_V = self.pooling(x=_V, edge_index=edge[[1, 0]], size=size[[1, 0]])

        x_out = _l_X + l_kinematics
        v_out = _l_V + l_kinematics - v

        return (x_out, v_out, h_out) if v is not None else (x_out, h_out)
