from torch import nn
import torch
import torch.nn.functional as F
from torch_sparse import spmm
from harl.utils.models_tools import init, get_init_method
import math
import numpy as np
from torch_geometric.utils import to_dense_adj


# theta = np.pi / 2
# rotation_matrix = torch.tensor([
#     [np.cos(theta), -np.sin(theta)], 
#     [np.sin(theta), np.cos(theta)]
# ], dtype=torch.float32)


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
        self.n_vector_input = n_vector_input
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input * n_vector_input // 3, self.activation, flat=flat)
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
        vec_scalar = vec_scalar.view(vec_scalar.shape[0], self.n_vector_input, -1)
        vector = torch.einsum('bij,bjk->bik', Z, vec_scalar)  # [N, 2]
        scalar = self.out_scalar_net(scalar)  # [N, H]

        return vector, scalar


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


class PoolingLayer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, n_vector_input, activation=nn.SiLU(), flat=False):
        super(PoolingLayer, self).__init__()
        self.edge_message_net = EquivariantEdgeScalarNet(
            n_vector_input=n_vector_input, 
            hidden_dim=hidden_nf,
            activation=activation, 
            n_scalar_input=2 * hidden_nf + in_edge_nf,
            norm=True, flat=flat
        )
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
        if edge_fea is None:
            hij = torch.cat((h[row], h[col]), dim=-1)
        else:
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
            layer = PoolingLayer(
                in_edge_nf, 
                hidden_nf, 
                n_vector_input=n_vector_input, 
                activation=activation, 
                flat=flat
            )
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
            # vectors = torch.stack(vectors, dim=-1)  # [BN, 2, V]
            vectors = torch.cat(vectors, dim=-1)  # [BN, 2, V]
        for i in range(self.n_layers):
            vectors, h = self.layers[i](vectors, h, edge_index, edge_fea)  # vectors: [BN, 2, V], h: [BN, H]
        pooling = self.pooling(h)  # [BN, C]  C is the number of clusters
        return pooling, vectors, h  # [BN, C]  此处没有返回vectors，不知是何原因，pooling将用于softmax计算聚类分数


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
        # self.input_dim = n_vector_input * n_vector_input + n_scalar_input
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
        scalar = torch.norm(Z, p=2, dim=1)
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, VV]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, VV + L] L=257
        scalar = self.scalar_net(scalar)  # [N, H]
        return scalar


class EGNNv2_Layer(nn.Module):
    def __init__(
        self, 
        in_edge_nf, 
        hidden_nf, 
        n_vector_input,
        activation=nn.SiLU(), 
        flat=False, 
        norm=False
    ):
        super(EGNNv2_Layer, self).__init__()
        self.edge_message_net = InvariantScalarNet(
            n_vector_input=n_vector_input, 
            hidden_dim=hidden_nf, 
            output_dim=hidden_nf,
            activation=activation, 
            n_scalar_input=2 * hidden_nf + in_edge_nf,
            norm=norm, 
            last_act=True, 
            flat=flat
        )
        self.n_vector_input = n_vector_input
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)
        self.equ_net = nn.Linear(n_vector_input, n_vector_input, bias=False)  # 可学习权重，用于不同通道之间进行混合
        self.node_equ_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                    flat=flat)


    def forward(self, equ, h, edge_index, edge_fea):
        """
        :param equ: 等变特征，[BN, n, M]
        :param h: 不变特征，[BN, H]
        """
        row, col = edge_index
        rij = equ[row] - equ[col]
        if edge_fea is not None:
            hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [B*90, 2H+1], hij is the concat of hi, hj and edge feature
        else:
            hij = torch.cat((h[row], h[col]), dim=-1)
        
        message = self.edge_message_net(vectors=rij, scalars=hij)  # [B*90, H], message is mij
        coord_message = self.coord_net(message)  # [B*90, 1]
        f = (equ[row] - equ[col]) * coord_message.unsqueeze(-1)  # [B*90, 2] update of mij
        f = f.reshape(f.shape[0], -1)
        tot_f = aggregate(message=f, row_index=row, n_node=equ.shape[0], aggr='mean')  # [BN, 2]
        tot_f = torch.clamp(tot_f, min=-100, max=100) # tot_f is used to update x
        tot_f = tot_f.reshape(tot_f.shape[0], -1, self.n_vector_input)

        # equ = equ + self.equ_net(tot_f)
        equ = self.node_equ_net(h).unsqueeze(-1) * equ + tot_f
        tot_message = aggregate(message=message, row_index=row, n_node=equ.shape[0], aggr='sum')  # [BN, H], sum of mij
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, 2H]
        h = self.node_net(node_message)  # [BN, H], update of h
        return equ, h


class EGNNv2(nn.Module):
    def __init__(
        self, 
        n_layers, 
        in_node_nf, 
        in_edge_nf, 
        hidden_nf, 
        n_vector_input,
        activation=nn.SiLU(), 
        device='cpu', 
        flat=False, 
        norm=False
    ):
        super(EGNNv2, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = EGNNv2_Layer(
                in_edge_nf, 
                hidden_nf, 
                n_vector_input,
                activation, 
                flat, 
                norm
            )
            self.layers.append(layer)
        self.to(device)

    def forward(self, equ, h, edge_index, edge_fea):
        h = self.embedding(h)   # [BN, H]
        for i in range(self.n_layers):
            equ, h = self.layers[i](equ, h, edge_index, edge_fea) # x: [BN, 2], v: [BN, 2], h: [BN, H]
        return equ, h


class EGAN_Layer(nn.Module):
    def __init__(
        self, 
        in_edge_nf, 
        hidden_nf, 
        n_vector_input,
        num_heads,
        embed_size,
        activation=nn.SiLU(), 
        flat=False, 
        norm=False
    ):
        super(EGAN_Layer, self).__init__()
        self.num_heads = num_heads   # 多头
        self.embed_size = embed_size # 注意力嵌入维度
        self.head_dim = embed_size // num_heads

        self.equ_encoder = nn.Linear(n_vector_input, embed_size, bias=False)
        self.g_proj = nn.Linear(embed_size, 32, bias=False)
        self.vg_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.linear_g1 = nn.Linear(32 * 32, embed_size * 2)
        self.linear_g2 = nn.Linear(embed_size * 2, embed_size)
        self.ng_out = nn.Linear(embed_size, embed_size)
        self.g_out = nn.Linear(embed_size, embed_size, bias=False)
        self.g_decoder = nn.Linear(embed_size, n_vector_input, bias=False)
        self.h_decoder = nn.Linear(embed_size, embed_size)

        self.query_net = nn.Linear(self.embed_size + self.embed_size, self.embed_size)
        self.key_net = nn.Linear(self.embed_size + self.embed_size, self.embed_size)
        self.value_net = nn.Linear(self.embed_size + self.embed_size, self.embed_size)
        # self.equ_value_net = nn.Linear(n_vector_input, self.embed_size, bias=False)
        # self.equ_out_net = nn.Linear(self.embed_size, n_vector_input, bias=False)

        # self.edge_message_net = InvariantScalarNet(
        #     n_vector_input=n_vector_input,
        #     hidden_dim=hidden_nf, 
        #     output_dim=hidden_nf,
        #     activation=activation, 
        #     n_scalar_input=hidden_nf,
        #     norm=norm, 
        #     last_act=True, 
        #     flat=flat
        # )

        self.n_vector_input = n_vector_input
        # self.node_net = BaseMLP(input_dim=self.embed_size, hidden_dim=hidden_nf, output_dim=hidden_nf,
        #                         activation=activation, flat=flat)

    def forward(self, equ, h, edge_index, edge_fea):
        """
        :param equ: 等变特征，[BN, n, M]
        :param h: 不变特征，[BN, H]
        """

        edge_index = torch.stack(edge_index)
        scaling = float(self.head_dim) ** -0.5
        n_node, n = equ.shape[:2]

        # theta = np.pi / 2
        # rotation_matrix = torch.tensor([
        #     [np.cos(theta), -np.sin(theta)], 
        #     [np.sin(theta), np.cos(theta)]
        # ], dtype=torch.float32)

        # rotation_matrix = rotation_matrix.expand(equ.shape[0], -1, -1).to(equ.device)
        # equ_r = torch.bmm(rotation_matrix, equ)
        
        g_src = self.equ_encoder(equ) * math.sqrt(self.embed_size)
        h_src = h * math.sqrt(self.embed_size)

        g_src2 = self.g_proj(g_src)
        g_src2 = torch.bmm(g_src2.transpose(-2, -1), g_src2)
        # F_norm = torch.norm(g_src2, dim=(-2, -1), keepdim=True) + 1.0
        # F_norm = F_norm.reshape(n_node, -1)
        g_src2 = g_src2.reshape(n_node, -1)
        g_src2 = self.linear_g2(F.relu(self.linear_g1(g_src2)))
        h_src2 = torch.cat([g_src2, h_src], dim=-1)

        # q = self.query_net(h_src2) / F_norm
        # k = self.key_net(h_src2) / F_norm
        # v = self.value_net(h_src2) / F_norm
        q = self.query_net(h_src2)
        k = self.key_net(h_src2)
        v = self.value_net(h_src2)
        vg = self.vg_proj(g_src)

        q = q * scaling

        q = q.reshape((n_node, self.num_heads, self.head_dim)).transpose(0, 1)
        k = k.reshape((n_node, self.num_heads, self.head_dim)).transpose(0, 1)
        v = v.reshape((n_node, self.num_heads, self.head_dim)).transpose(0, 1)
        
        vg = vg.reshape(n_node, n, self.num_heads, -1)
        vg = vg.permute(1, 2, 0, 3)
        vg = vg.reshape(-1, n_node, self.head_dim)

        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        # attn_output_weights_r = torch.bmm(qr, kr.transpose(-2, -1))
        # adj = to_dense_adj(edge_index)
        # adj = adj + torch.eye(n_node).to(adj.device)
        # attn_mask = (adj == 0).float() * -1e9
        # attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        # attn_output_weights = attn_output_weights + attn_mask
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(n_node, -1)
        h_src1 = self.ng_out(attn_output)

        attn_output_weights = attn_output_weights.repeat(n, 1, 1)
        attn_output = torch.bmm(attn_output_weights, vg)
        attn_output = attn_output.transpose(0, 1).reshape(n_node, n, self.num_heads, -1)
        attn_output = attn_output.reshape(n_node, n, -1)
        g_src1 = self.g_out(attn_output)

        g_src = g_src + g_src1
        h_src = h_src + h_src1
        # # message = self.edge_message_net(vectors=equ_embed, scalars=h)

        # # query = self.query_net(message).reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        # # key = self.query_net(message).reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        # # value = self.value_net(message).reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        # equ_value = self.equ_value_net(equ_embed).reshape(n_node, n, self.num_heads, -1)
        # equ_value = equ_value.permute(1, 2, 0, 3)
        # equ_value = equ_value.reshape(-1, n_node, self.head_dim)

        # attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # adj = to_dense_adj(torch.stack(edge_index))
        # attn_mask = adj.repeat(self.num_heads, 1, 1)
        # attention = attention.masked_fill(attn_mask == 0, -1e9)
        # score = nn.Softmax(dim=-1)(attention)

        # attn_h = score @ value
        # attn_h = attn_h.transpose(0, 1).reshape(h.shape[0], -1)

        # score = score.repeat(n, 1, 1)
        # attn_equ = (score @ equ_value)
        # attn_equ = attn_equ.transpose(0, 1).reshape(n_node, n, self.num_heads, -1)
        # attn_equ = attn_equ.transpose(1, 2).reshape(n_node, n, -1)
        # attn_equ_out = self.equ_out_net(attn_equ)

        # equ_out = equ + attn_equ_out

        # h = h + self.node_net(attn_h)  # [BN, H], update of h
        equ_out = self.g_decoder(g_src)
        h_out = self.h_decoder(h_src)
        return equ_out, h_out


class EGAN(nn.Module):
    def __init__(
        self, 
        n_layers, 
        in_node_nf, 
        in_edge_nf, 
        hidden_nf, 
        n_vector_input,
        num_heads,
        embed_size,
        activation=nn.SiLU(), 
        device='cpu', 
        flat=False, 
        norm=False
    ):
        super(EGAN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        assert embed_size % num_heads == 0, "Attention Embedding size should be divisible by number of heads."

        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = EGAN_Layer(
                in_edge_nf, 
                hidden_nf, 
                n_vector_input,
                num_heads,
                embed_size,
                activation, 
                flat, 
                norm
            )
            self.layers.append(layer)
        self.device = device
        self.to(device)

    def forward(self, equ, h, edge_index, edge_fea):
        h = self.embedding(h)   # [BN, H]

        # theta = np.pi / 2
        # rotation_matrix = torch.tensor([
        #     [np.cos(theta), -np.sin(theta)], 
        #     [np.sin(theta), np.cos(theta)]
        # ], dtype=torch.float32)
        # rotation_matrix = rotation_matrix.expand(equ.shape[0], -1, -1).to(self.device)
        # equ_r = torch.bmm(rotation_matrix, equ)

        for i in range(self.n_layers):
            equ, h = self.layers[i](equ, h, edge_index, edge_fea)
            # equ_r, h_r = self.layers[i](equ_r, h, edge_index, edge_fea)
            # equ_out_r = torch.bmm(rotation_matrix, equ_out)

        return equ, h


class EGHN(nn.Module):
    def __init__(
        self, 
        in_node_nf, 
        in_edge_nf, 
        hidden_nf, 
        n_cluster, 
        n_vector_input,
        layer_per_block=3, 
        layer_pooling=3, 
        layer_decoder=1, 
        flat=False, 
        activation=nn.SiLU(), 
        device='cpu', 
        norm=False
    ):
        super(EGHN, self).__init__()
        node_hidden_dim = hidden_nf
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.current_pooling_plan = None
        self.n_cluster = n_cluster  # 4 for simulation and 5 for mocap
        self.n_vector_input = n_vector_input
        self.n_layer_per_block = layer_per_block
        self.n_layer_pooling = layer_pooling
        self.n_layer_decoder = layer_decoder
        self.flat = flat

        # low-level force net
        self.low_force_net = EGNNv2(
            n_layers=self.n_layer_per_block,
            in_node_nf=hidden_nf, 
            in_edge_nf=in_edge_nf, 
            hidden_nf=hidden_nf,
            n_vector_input=n_vector_input,
            activation=activation, 
            device=device, 
            flat=flat, 
            norm=norm
        )
        # self.low_force_net = EGAN(
        #     n_layers=self.n_layer_per_block,
        #     in_node_nf=hidden_nf, 
        #     in_edge_nf=in_edge_nf, 
        #     hidden_nf=hidden_nf,
        #     n_vector_input=n_vector_input,
        #     num_heads=1,
        #     embed_size=128,
        #     activation=activation, 
        #     device=device, 
        #     flat=flat, 
        #     norm=norm
        # )
        self.low_pooling = PoolingNet(
            n_vector_input=n_vector_input * 2, 
            hidden_nf=hidden_nf, 
            output_nf=self.n_cluster,
            activation=activation, 
            in_edge_nf=in_edge_nf, 
            n_layers=self.n_layer_pooling, 
            flat=flat
        )
        self.high_force_net = EGNNv2(
            n_layers=self.n_layer_per_block,
            in_node_nf=hidden_nf, 
            in_edge_nf=1, 
            hidden_nf=hidden_nf,
            n_vector_input=n_vector_input,                      
            activation=activation, 
            device=device, 
            flat=flat
        )
        # self.high_force_net = EGAN(
        #     n_layers=self.n_layer_per_block,
        #     in_node_nf=hidden_nf, 
        #     in_edge_nf=1, 
        #     hidden_nf=hidden_nf,
        #     n_vector_input=n_vector_input,  
        #     num_heads=1,
        #     embed_size=128,                    
        #     activation=activation, 
        #     device=device, 
        #     flat=flat
        # )
        if self.n_layer_decoder == 1:
            self.kinematics_net = EquivariantScalarNet(
                n_vector_input=n_vector_input * 3,
                hidden_dim=hidden_nf,
                activation=activation,
                n_scalar_input=node_hidden_dim + node_hidden_dim,
                norm=True,
                flat=flat
            )
        else:
            self.kinematics_net = EGMN(
                n_vector_input=n_vector_input * 3, 
                hidden_dim=hidden_nf, 
                activation=activation,
                n_scalar_input=node_hidden_dim + node_hidden_dim,
                norm=True, 
                flat=flat,
                n_layers=self.n_layer_decoder
            )

        self.to(device)

    def forward(self, equ, h, edge_index, edge_fea, local_edge_index, local_edge_fea, n_node, flag, node_mask=None, node_nums=None):
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
        n = equ.shape[1]  # 维度
        h = self.embedding(h)  # [BN, 36] -> [BN, H]
        row, col = edge_index  # 全局观测下维度[B * 90]，完全图；局部观测下根据智能体距离构建邻接矩阵

        ''' low level force '''
        new_equ, h = self.low_force_net(equ, h, edge_index, edge_fea)  # new_x: [BN, 2], new_v: [BN, 2], h: [BN, H]

        nf_equ = new_equ - equ

        ''' pooling network '''
        pooling_fea, vctors, h_ = self.low_pooling(vectors=[equ, nf_equ], h=h,
                                       edge_index=local_edge_index, edge_fea=local_edge_fea)  # [BN, C] C is the number of clusters
        '''hard_pooling 用于显示聚类结果'''
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
        p_index = torch.ones_like(equ[:, :, 0])[..., 0]  # [BN, ]
        if node_mask is not None:
            p_index = p_index * node_mask
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, C, 1]  对st的行求和

        _equ = equ.reshape(-1, n_node, equ.shape[-2] * equ.shape[-1])
        _h = h.reshape(-1, n_node, h.shape[-1])
        _nf_equ = nf_equ.reshape(-1, n_node, nf_equ.shape[-2] * nf_equ.shape[-1])

        EQU = torch.einsum('bij,bjk->bik', sT, _equ)
        H = torch.einsum('bij,bjk->bik', sT, _h)
        NF_EQU = torch.einsum('bij,bjk->bik', sT, _nf_equ)

        EQU, H, NF_EQU = EQU / count, H / count, NF_EQU / count

        EQU = EQU.reshape(-1, n, self.n_vector_input)
        H = H.reshape(-1, H.shape[-1])
        NF_EQU = NF_EQU.reshape(-1, n, self.n_vector_input)

        a = spmm(torch.stack((local_edge_index[0], local_edge_index[1]), dim=0),
                 torch.ones_like(local_edge_index[0]), equ.shape[0], equ.shape[0], pooling)  # [BN, C]
        a = a.reshape(-1, n_node, a.shape[-1])  # [B, N, C]
        '''
        A is constructed by connectivity-based local edges
        AA is constructed by distance-based global edges
        '''
        A = torch.einsum('bij,bjk->bik', sT, a)  # [B, C, C]
        # self.cut_loss = self.get_cut_loss(A)
        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), equ.shape[0], equ.shape[0], pooling)  # [BN, C]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, C]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, C, C]

        # construct high-level edges
        h_row, h_col, h_edge_fea, h_edge_mask = self.construct_edges(AA, AA.shape[-1])  # [BCC]
        ''' high-level message passing '''
        h_new_equ, h_new_h = self.high_force_net(EQU, H, (h_row, h_col), h_edge_fea.unsqueeze(-1))
        h_nf_equ = h_new_equ - EQU

        ''' high-level kinematics update '''
        _EQU = h_new_equ
        _H = h_new_h  # [BC, H]

        ''' low-level kinematics update '''

        l_nf_equ = h_nf_equ.reshape(-1, AA.shape[1], n * self.n_vector_input)
        l_nf_equ = torch.einsum('bij,bjk->bik', s, l_nf_equ).reshape(-1, n, self.n_vector_input)

        l_EQU = EQU.reshape(-1, AA.shape[1], n * self.n_vector_input)
        l_EQU = torch.einsum('bij,bjk->bik', s, l_EQU).reshape(-1, n, self.n_vector_input)

        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, C, H]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, H]

        l_kinematics, h_out = self.kinematics_net(vectors=torch.cat((l_nf_equ, equ - l_EQU, nf_equ), dim=-1), 
                                                  scalars=torch.cat((h, l_H), dim=-1))  # [BN, 2], [BN, H]

        _l_EQU = _EQU.reshape(-1, AA.shape[1], n * self.n_vector_input)
        _l_EQU = torch.einsum('bij,bjk->bik', s, _l_EQU).reshape(-1, n, self.n_vector_input)

        equ_out = _l_EQU + l_kinematics

        return (equ_out, h_out)

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


