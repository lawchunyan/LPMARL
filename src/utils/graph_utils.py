from functools import partial


def filter_node_idx_func(nodes, ntype_idx):
    return nodes.data['node_type'] == ntype_idx


def get_filtered_node_idx(graph, ntype_idx):
    node_filter_func = partial(filter_node_idx_func, ntype_idx=ntype_idx)
    node_idx = graph.filter_nodes(node_filter_func)
    return node_idx


def filter_edge_idx_func(edges, etype_idx):
    return edges.data['edge_type'] == etype_idx


def get_filtered_edge_idx(graph, etype_idx):
    edge_filter_func = partial(filter_edge_idx_func, etype_idx=etype_idx)
    edge_idx = graph.filter_edges(edge_filter_func)
    return edge_idx
