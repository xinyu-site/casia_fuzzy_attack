import os
import copy
import pickle
import itertools
import numpy as np
import networkx as nx
from harl.utils.encoding_tree import PartitionTree
import torch

def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def pool_trans(input_):
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
    return g


def pool_trans_discon(input_):
    g, tree_depth = input_
    if nx.is_connected(g['G']):
        return pool_trans((g, tree_depth))
    trees = []
    for gi, sub_nodes in enumerate(nx.connected_components(g['G'])):
        if len(sub_nodes) == 1:
            node = list(sub_nodes)[0]
            js = [{'ID': node, 'parent': '%s_%s_0' % (gi, 1), 'depth': 0, 'children': None}]
            for d in range(1, tree_depth+1):
                js.append({'ID': '%s_%s_0' % (gi, d),
                           'parent': '%s_%s_0' % (gi, d+1) if d<tree_depth else None,
                           'depth': d,
                           'children': [js[-1]['ID']]
                          })
        else:
            sg = g['G'].subgraph(sub_nodes)
            nodes = list(sg.nodes)
            nodes.sort()
            nmap = {n: nodes.index(n) for n in nodes}
            sg = nx.relabel_nodes(sg, nmap)
            adj_mat = trans_to_adj(sg)
            tree = trans_to_tree(adj_mat, tree_depth)
            tree = update_node(tree)
            js = list(tree.values())
            rmap = {nodes.index(n): n for n in nodes}
            for j in js:
                if j['depth'] > 0:
                    rmap[j['ID']] = '%s_%s_%s' % (gi, j['depth'], j['ID'])
            for j in js:
                j['ID'] = rmap[j['ID']]
                j['parent'] = rmap[j['parent']] if j['depth']<tree_depth else None
                j['children'] = [rmap[c] for c in j['children']] if j['children'] else None
        trees.append(js)
    id_map = {}
    for d in range(0, tree_depth+1):
        for js in trees:
            for j in js:
                if j['depth'] == d:
                    id_map[j['ID']] = len(id_map) if d>0 else j['ID']
    tree = {}
    root_ids = []
    for js in trees:
        for j in js:
            n = copy.deepcopy(j)
            n['parent'] = id_map[n['parent']] if n['parent'] else None
            n['children'] = [id_map[c] for c in n['children']] if n['children'] else None
            n['ID'] = id_map[n['ID']]
            tree[n['ID']] = n
            if n['parent'] is None:
                root_ids.append(n['ID'])
    root_id = min(root_ids)
    root_children = list(itertools.chain.from_iterable([tree[i]['children'] for i in root_ids]))
    root_node = {'ID': root_id, 'parent': None, 'children': root_children, 'depth': tree_depth}
    [tree.pop(i) for i in root_ids]
    for c in root_children:
        tree[c]['parent'] = root_id
    tree[root_id] = root_node
    g['tree'] = tree
    return g


def trans(input_):
    g, tree_depth = input_
    adj_mat = trans_to_adj(g)
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    return tree


def trans_discon(input_):
    g, tree_depth = input_
    if nx.is_connected(g):
        return trans((g, tree_depth))
    trees = []
    for gi, sub_nodes in enumerate(nx.connected_components(g)):
        if len(sub_nodes) == 1:
            node = list(sub_nodes)[0]
            js = [{'ID': node, 'parent': '%s_%s_0' % (gi, 1), 'depth': 0, 'children': None}]
            for d in range(1, tree_depth+1):
                js.append({'ID': '%s_%s_0' % (gi, d),
                           'parent': '%s_%s_0' % (gi, d+1) if d<tree_depth else None,
                           'depth': d,
                           'children': [js[-1]['ID']]
                          })
        else:
            sg = g.subgraph(sub_nodes)
            nodes = list(sg.nodes)
            nodes.sort()
            nmap = {n: nodes.index(n) for n in nodes}
            sg = nx.relabel_nodes(sg, nmap)
            adj_mat = trans_to_adj(sg)
            tree = trans_to_tree(adj_mat, tree_depth)
            tree = update_node(tree)
            js = list(tree.values())
            rmap = {nodes.index(n): n for n in nodes}
            for j in js:
                if j['depth'] > 0:
                    rmap[j['ID']] = '%s_%s_%s' % (gi, j['depth'], j['ID'])
            for j in js:
                j['ID'] = rmap[j['ID']]
                j['parent'] = rmap[j['parent']] if j['depth']<tree_depth else None
                j['children'] = [rmap[c] for c in j['children']] if j['children'] else None
        trees.append(js)
    id_map = {}
    for d in range(0, tree_depth+1):
        for js in trees:
            for j in js:
                if j['depth'] == d:
                    id_map[j['ID']] = len(id_map) if d>0 else j['ID']
    tree = {}
    root_ids = []
    for js in trees:
        for j in js:
            n = copy.deepcopy(j)
            n['parent'] = id_map[n['parent']] if n['parent'] else None
            n['children'] = [id_map[c] for c in n['children']] if n['children'] else None
            n['ID'] = id_map[n['ID']]
            tree[n['ID']] = n
            if n['parent'] is None:
                root_ids.append(n['ID'])
    root_id = min(root_ids)
    root_children = list(itertools.chain.from_iterable([tree[i]['children'] for i in root_ids]))
    root_node = {'ID': root_id, 'parent': None, 'children': root_children, 'depth': tree_depth}
    [tree.pop(i) for i in root_ids]
    for c in root_children:
        tree[c]['parent'] = root_id
    tree[root_id] = root_node
    # g['tree'] = tree
    return tree


def get_layer_graph(tree, graph, tree_depth):
    layer_graph = [graph]
    for l in range(1, tree_depth):
        partition = {frozenset([tree[c].get('graphID', c) for c in n['children']]): i
                     for i, n in tree.items() if n['depth']==l}
        lg = nx.quotient_graph(layer_graph[-1], partition.keys(), relabel=False)
        lg = nx.relabel_nodes(lg, partition)
        layer_graph.append(lg)
    return layer_graph


def extract_tree(graph, tree_depth):
    leaf_size = len(graph['G'])
    tree = {'node_size': [0] * (tree_depth+1),
            'edges': [[] for i in range(tree_depth+1)],
            'node_degrees': [0] * leaf_size,
            }
    old_tree = copy.deepcopy(graph['tree'])
    # tree layer mask
    layer_idx = [0]  # 记录每个层的起始结点索引
    for layer in range(tree_depth + 1):
        layer_nodes = [i for i, n in old_tree.items() if n['depth'] == layer]  # 该层结点索引
        layer_idx.append(layer_idx[-1] + len(layer_nodes))  # 记录下一层的起始结点索引
        tree['node_size'][layer] = len(layer_nodes)  # 记录该层结点数

    for i, n in old_tree.items():
        # edge
        if n['depth'] > 0:
            n_idx = n['ID'] - layer_idx[n['depth']]  # 该结点在该层的索引
            c_base = layer_idx[n['depth'] - 1]  # 该层子结点的起始索引
            tree['edges'][n['depth']].extend([(n_idx, c-c_base) for c in n['children']])  # 将子结点索引与该节点索引组成边
            continue
        # leaf: node feature
        graphID = n.get('graphID', n['ID'])
        nid = n['ID']
        tree['node_degrees'][nid] = graph['G'].degree[graphID]

    # for gin
    layer_graphs = get_layer_graph(old_tree, graph['G'], tree_depth)
    layer_edgeMat = []
    for l in range(tree_depth):
        g = layer_graphs[l]
        nmap = {n: n-layer_idx[l] for n in g.nodes}
        g = nx.relabel_nodes(g, nmap)
        edges = [[n1, n2] for n1, n2 in g.edges]
        edges.extend([[n2, n1] for n1, n2 in edges])
        edge_mat = torch.LongTensor(edges).transpose(0, 1)
        layer_edgeMat.append(edge_mat)
    tree['graph_mats'] = layer_edgeMat  # 记录每层图的边矩阵
    return tree


def trans_graph_tree(G, tree_depth):
    # G: networkx.Graph
    # tree_depth: 2-n, not including the leaf layer
    # return dict: {nodeID: {parent: nodeID, children: [], depth: int}}
    adj_mat = trans_to_adj(G)
    tree = trans_to_tree(adj_mat, tree_depth)
    return update_node(tree)


def extract_layer_data(T, G, tree_depth):
    node_size = [0] * (tree_depth+1)
    # node size and layer index base
    layer_idx = [0]
    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in T.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
        node_size[layer] = len(layer_nodes)

    interLayerEdges = [[] for i in range(tree_depth+1)]
    # edges among layers
    for i, n in T.items():
        if n['depth'] == 0:
            continue
        n_idx = n['ID'] - layer_idx[n['depth']]
        c_base = layer_idx[n['depth']-1]
        interLayerEdges[n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
    interLayer_edgeMat = [torch.LongTensor(es).T for es in interLayerEdges]

    # for gnn
    layer_graphs = get_layer_graph(T, G, tree_depth)
    layer_edgeMat = []
    layer_edgeWeight = []
    for l in range(tree_depth):
        g = layer_graphs[l]
        nmap = {n: n-layer_idx[l] for n in g.nodes}
        g = nx.relabel_nodes(g, nmap)
        edges = [[n1, n2] for n1, n2 in g.edges]
        edges.extend([[n2, n1] for n1, n2 in edges])
        edge_mat = torch.LongTensor(edges).T
        layer_edgeMat.append(edge_mat)
        weights = [g[n1][n2].get('weight', 1.0) for n1, n2 in g.edges]
        weights = weights + weights  # bidirectional
        layer_edgeWeight.append(torch.tensor(weights, dtype=torch.float))

    return {'node_size': node_size,
            'interLayer_edgeMat': interLayer_edgeMat,
            'layer_edgeMat': layer_edgeMat,
            'layer_edgeWeight': layer_edgeWeight,
            }