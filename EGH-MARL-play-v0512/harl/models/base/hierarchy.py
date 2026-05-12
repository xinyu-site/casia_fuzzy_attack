from torch import nn
import torch
import torch.nn.functional as F
from torch_sparse import spmm
from harl.utils.models_tools import init, get_active_func, get_init_method
from torch_geometric.nn import SAGEConv, GATv2Conv


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
                init_(nn.Linear(input_dim, hidden_dim)),
                activation,
                # nn.LayerNorm(hidden_dim),
                init_(nn.Linear(hidden_dim, output_dim)),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)),
                activation,
                # nn.LayerNorm(hidden_dim),    # add layer norm
                init_(nn.Linear(hidden_dim, output_dim))
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class Hierarchy(nn.Module):
    def __init__(self, in_node_nf, out_nf, hidden_nf, n_cluster, n_layer, use_res, device='cpu'):
        super(Hierarchy, self).__init__()
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.n_cluster = n_cluster
        self.n_layers = n_layer
        self.use_res = use_res

        # low-level force net
        for i in range(0, self.n_layers):
            self.add_module("low_level_sage_%d" % i, SAGEConv(hidden_nf, hidden_nf))   
            self.add_module("high_level_sage_%d" % i, SAGEConv(hidden_nf, hidden_nf))
        
        # 用于smacv2
        self.low_force_net = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), last_act=True, flat=True)
        self.high_force_net = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), flat=True)
        
        self.low_pooling = nn.Linear(hidden_nf, self.n_cluster)
        self.mlp = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), residual=True, flat=True)

        self.to(device)

    def forward(self, obs, edge_index, n_node):
        h = self.embedding(obs)
        hh = h  # 记录原始h
        row, col = edge_index
        edge_index = torch.stack(edge_index)

        ''' low level force '''
        # h_ = self.low_force_net(h, edge_index)
        # h_res = h  # 保留原始输入
        # for j in range(0, self.n_layers):
        #     h = self._modules["low_level_sage_%d" % j](h, edge_index)
        #     if j != self.n_layers - 1:
        #         h = torch.tanh(h)
        # if self.use_res:
        #     h = h + h_res  # 应用残差连接
        h = self.low_force_net(h)

        ''' pooling network '''
        pooling_fea = self.low_pooling(h)  # [BN, C] C is the number of clusters
        '''hard_pooling 用于显示聚类结果'''
        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)
        self.current_pooling_plan = hard_pooling  # record the pooling plan
        # self.inspect_pooling_plan()

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, C]

        sT = s.transpose(-2, -1)  # [B, C, N]
        p_index = torch.ones_like(h)[..., 0]  # [BN, ]
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, C, 1]  对st的行求和
        _h = h.reshape(-1, n_node, h.shape[-1])

        H = torch.einsum('bij,bjk->bik', sT, _h)
        H = H / count
        # X: [B, C, 2], H: [B, C, H], NF: [B, C, 2], V: [B, C, 2]
        H = H.reshape(-1, H.shape[-1])
    
        '''
        A is constructed by connectivity-based local edges
        AA is constructed by distance-based global edges
        '''

        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), h.shape[0], h.shape[0], pooling)  # [BN, C]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, C]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, C, C]

        # construct high-level edges
        h_row, h_col, _, _ = self.construct_edges(AA, AA.shape[-1])  # [BCC]
        
        ''' high-level message passing '''
        # h_new_h = self.high_force_net(H, torch.stack([h_row, h_col]))  # h_new_x, h_new_v: [BC, 2], h_new_h: [BC, H]
        # H_res = H  # 保留原始输入
        # for j in range(0, self.n_layers):
        #     H = self._modules["high_level_sage_%d" % j](H, torch.stack([h_row, h_col]))
        #     if j != self.n_layers - 1:
        #         H = torch.tanh(H)
        # if self.use_res:
        #     H = H + H_res  # 应用残差连接
        H = self.high_force_net(H)
    
        ''' high-level kinematics update '''
        _H = H  # [BC, H]

        ''' low-level kinematics update '''
        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, C, H]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, H]

        # out = self.kinematics_net(l_H)  # [BN, 2], [BN, H]
        l_H = self.mlp(l_H + hh)

        # return out, l_H
        return l_H

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


class HAMA(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, n_cluster, n_layer, use_res, n_heads=4, device='cpu'):
        super(HAMA, self).__init__()
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.n_cluster = n_cluster
        self.n_layers = n_layer
        self.use_res = use_res
        self.n_heads = n_heads

        # low/high-level message passing
        self.low_level_gats = nn.ModuleList(
            [GATv2Conv(hidden_nf, hidden_nf, heads=self.n_heads, concat=False, dropout=0.1)
             for _ in range(self.n_layers)]
        )
        self.high_level_gats = nn.ModuleList(
            [GATv2Conv(hidden_nf, hidden_nf, heads=self.n_heads, concat=False, dropout=0.1)
             for _ in range(self.n_layers)]
        )
        self.low_norms = nn.ModuleList([nn.LayerNorm(hidden_nf) for _ in range(self.n_layers)])
        self.high_norms = nn.ModuleList([nn.LayerNorm(hidden_nf) for _ in range(self.n_layers)])
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)
        
        self.low_pooling = nn.Linear(hidden_nf, self.n_cluster)
        self.pooling_dropout = nn.Dropout(p=0.1)
        self.mlp = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), residual=True, flat=True)

        self.to(device)

    def forward(self, obs, edge_index, n_node):
        h = self.embedding(obs)
        hh = h  # 记录原始h
        row, col = edge_index
        edge_index = torch.stack(edge_index) if not torch.is_tensor(edge_index) else edge_index
        if edge_index.device != h.device:
            edge_index = edge_index.to(h.device)

        ''' low level force '''
        h_res = h if self.use_res else None  # 保留原始输入
        for j, layer in enumerate(self.low_level_gats):
            h_in = h
            h = layer(h, edge_index)
            h = self.low_norms[j](h)
            if j != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            if self.use_res:
                h = h + h_in
        if self.use_res:
            h = h + h_res  # 应用残差连接

        ''' pooling network '''
        pooling_fea = self.pooling_dropout(self.low_pooling(h))  # [BN, C] C is the number of clusters
        '''hard_pooling 用于显示聚类结果'''
        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)
        self.current_pooling_plan = hard_pooling  # record the pooling plan

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, C]

        sT = s.transpose(-2, -1)  # [B, C, N]
        p_index = torch.ones_like(h)[..., 0]  # [BN, ]
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, C, 1]  对st的行求和
        _h = h.reshape(-1, n_node, h.shape[-1])

        H = torch.einsum('bij,bjk->bik', sT, _h)
        H = H / count
        # X: [B, C, 2], H: [B, C, H], NF: [B, C, 2], V: [B, C, 2]
        H = H.reshape(-1, H.shape[-1])
    
        '''
        A is constructed by connectivity-based local edges
        AA is constructed by distance-based global edges
        '''

        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), h.shape[0], h.shape[0], pooling)  # [BN, C]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, C]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, C, C]

        # construct high-level edges
        h_row, h_col, _, _ = self.construct_edges(AA, AA.shape[-1])  # [BCC]
        
        ''' high-level message passing '''
        H_res = H if self.use_res else None  # 保留原始输入
        h_edge_index = torch.stack([h_row, h_col])
        for j, layer in enumerate(self.high_level_gats):
            H_in = H
            H = layer(H, h_edge_index)
            H = self.high_norms[j](H)
            if j != self.n_layers - 1:
                H = self.activation(H)
                H = self.dropout(H)
            if self.use_res:
                H = H + H_in
        if self.use_res:
            H = H + H_res  # 应用残差连接
    
        ''' high-level kinematics update '''
        _H = H  # [BC, H]

        ''' low-level kinematics update '''
        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, C, H]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, H]

        l_H = self.mlp(l_H + hh)

        return l_H

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
    

class HMF(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        n_cluster,
        n_layer,
        use_res,
        attn_heads=4,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        attn_ffn_hidden_scale=1.0,
        device='cpu',
    ):
        super(HMF, self).__init__()
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.n_cluster = n_cluster
        self.n_layers = n_layer
        self.use_res = use_res
        
        # low-level attention (no GNN)
        attn_heads = attn_heads if hidden_nf % attn_heads == 0 else 1
        self.low_attn_heads = attn_heads
        self.low_attn = nn.MultiheadAttention(
            embed_dim=hidden_nf,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.low_attn_norm = nn.LayerNorm(hidden_nf)
        ffn_hidden = max(int(hidden_nf * attn_ffn_hidden_scale), hidden_nf)
        self.low_attn_ffn = BaseMLP(hidden_nf, ffn_hidden, hidden_nf, nn.SiLU(), residual=False, flat=False)
        self.low_attn_ffn_norm = nn.LayerNorm(hidden_nf)
        self.low_attn_dropout = nn.Dropout(p=attn_dropout)
        self.low_attn_ffn_dropout = nn.Dropout(p=ffn_dropout)

        self.low_force_net = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), last_act=True, flat=True)
        self.high_force_net = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), flat=True)
        
        self.low_pooling = nn.Linear(hidden_nf, self.n_cluster)
        self.mlp = BaseMLP(hidden_nf, hidden_nf, hidden_nf, nn.SiLU(), residual=True, flat=True)

        self.to(device)

    def forward(self, obs, n_node):
        h = self.embedding(obs)
        hh = h  # 记录原始h
        # row, col = edge_index
        # edge_index = torch.stack(edge_index)

        ''' low level attention '''
        h = h.reshape(-1, n_node, h.shape[-1])  # [B, N, H]
        h_attn, _ = self.low_attn(h, h, h, need_weights=False)
        h = h + h_attn if self.use_res else h_attn
        h = self.low_attn_norm(h)
        h_ffn = self.low_attn_ffn(h.reshape(-1, h.shape[-1])).reshape(-1, n_node, h.shape[-1])
        h = h + h_ffn if self.use_res else h_ffn
        h = self.low_attn_ffn_norm(h)
        h = h.reshape(-1, h.shape[-1])

        ''' low level force (feature refinement) '''
        h = self.low_force_net(h)

        ''' pooling network '''
        pooling_fea = self.low_pooling(h)  # [BN, C] C is the number of clusters
        '''hard_pooling 用于显示聚类结果'''
        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, C]

        sT = s.transpose(-2, -1)  # [B, C, N]
        p_index = torch.ones_like(h)[..., 0]  # [BN, ]
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, C, 1]  对st的行求和
        _h = h.reshape(-1, n_node, h.shape[-1])

        H = torch.einsum('bij,bjk->bik', sT, _h)
        H = H / count
        # X: [B, C, 2], H: [B, C, H], NF: [B, C, 2], V: [B, C, 2]
        H = H.reshape(-1, H.shape[-1])
    
        '''
        A is constructed by connectivity-based local edges
        AA is constructed by distance-based global edges
        '''

        # aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), h.shape[0], h.shape[0], pooling)  # [BN, C]
        # aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, C]
        # AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, C, C]
        
        ''' high-level message passing '''
        H = self.high_force_net(H)
    
        ''' high-level kinematics update '''
        _H = H  # [BC, H]

        ''' low-level kinematics update '''
        l_H = _H.reshape(-1, self.n_cluster, _H.shape[-1])  # [B, C, H]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, H]

        l_H = self.mlp(l_H + hh)

        return l_H
