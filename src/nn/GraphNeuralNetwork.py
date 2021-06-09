import dgl
import torch
import torch.nn as nn

from functools import partial
from src.nn.MultiLayeredPerceptron import MultiLayeredPerceptron as MLP
from src.utils.graph_utils import get_filtered_node_idx, get_filtered_edge_idx


class GraphNeuralNet(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, hidden_dims=[32, 32], edge_types=[0, 1], node_type=0):
        super(GraphNeuralNet, self).__init__()

        ins = [in_node_dim] + hidden_dims
        outs = hidden_dims + [out_node_dim]

        self.layers = nn.ModuleList()

        for _in, _ou in zip(ins, outs):
            self.layers.append(GraphNeuralNetLayer(_in, _ou, edge_types=edge_types, node_type=node_type))

    def forward(self, g, nf):
        for l in self.layers:
            nf = l(g, nf)
        return nf


class GraphNeuralNetLayer(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, hidden_dim=32, edge_types=[0, 1], node_type=0):
        super(GraphNeuralNetLayer, self).__init__()

        self.edge_update_func = nn.ModuleDict()
        self.node_update_func = nn.ModuleDict()

        for e in edge_types:
            self.edge_update_func["{}".format(e)] = MLP(in_node_dim * 2, hidden_dim)

        self.node_update_func = MLP(hidden_dim * len(edge_types) + in_node_dim, out_node_dim)

        self.edge_types = edge_types
        self.node_type = node_type

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf

        for e in self.edge_types:
            target_edges = get_filtered_edge_idx(g, e)
            message_func = partial(self.message_func, etype=e)
            reduce_func = partial(self.reduce_func, etype=e)
            g.send_and_recv(target_edges, message_func, reduce_func)

        target_nodes = get_filtered_node_idx(g, ntype_idx=self.node_type)
        g.apply_nodes(self.apply_node_func, target_nodes)

        updated_nf = g.ndata.pop('updated_nf')
        return updated_nf

    def message_func(self, edges, etype):
        src_nf = edges.src['nf']
        dst_nf = edges.dst['nf']
        ef_input = torch.cat([src_nf, dst_nf], dim=-1)
        ef_output = self.edge_update_func["{}".format(etype)](ef_input)
        return {'message': ef_output}

    @staticmethod
    def reduce_func(nodes, etype):
        messages = nodes.mailbox['message']
        reduced_message = messages.mean(1)
        return {'message_{}'.format(etype): reduced_message}

    def apply_node_func(self, nodes):
        nf_update_inputs = []
        for e in self.edge_types:
            msg = nodes.data['message_{}'.format(e)]
            nf_update_inputs.append(msg)
        nf_update_inputs.append(nodes.data['nf'])
        nf_update_input = torch.cat(nf_update_inputs, dim=-1)
        updated_nf = self.node_update_func(nf_update_input)

        return {'updated_nf': updated_nf}


if __name__ == '__main__':
    n_nodes = 10
    n_edges = 10

    nf_dim = 17
    out_nf_dim = 7

    g = dgl.rand_graph(n_nodes, n_edges)

    node_feature = torch.rand((n_nodes, nf_dim))
    g.ndata['node_type'] = torch.randint(2, size=(n_nodes,))
    g.edata['edge_type'] = torch.randint(2, size=(n_edges,))

    gnn = GraphNeuralNet(nf_dim, out_nf_dim, hidden_dims=[32])

    print(gnn(g, node_feature).shape)
