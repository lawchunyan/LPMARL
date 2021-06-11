import dgl
import torch
import torch.nn as nn

from functools import partial
from src.nn.MultiLayeredPerceptron import MultiLayeredPerceptron as MLP
from src.utils.graph_utils import get_filtered_node_idx, get_filtered_edge_idx
from src.utils.torch_util import dn


class GraphNeuralNet(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, hidden_dims=[64], edge_types=[0, 1, 2], node_types=[0, 1]):
        super(GraphNeuralNet, self).__init__()

        ins = [in_node_dim] + hidden_dims
        outs = hidden_dims + [out_node_dim]

        self.layers = nn.ModuleList()

        for _in, _ou in zip(ins, outs):
            self.layers.append(GraphNeuralNetLayer(_in, _ou, edge_types=edge_types, node_types=node_types))

    def forward(self, g, nf):
        for l in self.layers:
            nf = l(g, nf)
        return nf


class GraphNeuralNetLayer(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, hidden_dim=32, edge_types=[0, 1, 2], node_types=[0, 1]):
        super(GraphNeuralNetLayer, self).__init__()

        self.edge_update_func = nn.ModuleDict()
        self.node_update_func = nn.ModuleDict()

        for e in edge_types:
            self.edge_update_func["{}".format(e)] = MLP(in_node_dim * 2, hidden_dim)

        for n in node_types:
            self.node_update_func["{}".format(n)] = MLP(hidden_dim * len(edge_types), out_node_dim)
            # self.node_update_func = MLP(hidden_dim * len(edge_types), out_node_dim)

        self.edge_types = edge_types
        self.node_types = node_types

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf

        for e in self.edge_types:
            target_edges = get_filtered_edge_idx(g, e)
            message_func = partial(self.message_func, etype=e)
            reduce_func = partial(self.reduce_func, etype=e)
            g.send_and_recv(target_edges, message_func, reduce_func)

        for n in self.node_types:
            apply_func = partial(self.apply_node_func, ntype=n)
            target_nodes = get_filtered_node_idx(g, ntype_idx=n)
            g.apply_nodes(apply_func, target_nodes)

        updated_nf = g.ndata.pop('updated_nf')

        g.ndata.pop('nf')
        g.ndata.pop('message_0')
        g.ndata.pop('message_1')
        g.ndata.pop('message_2')

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

    def apply_node_func(self, nodes, ntype):
        nf_update_inputs = []
        for e in self.edge_types:
            msg = nodes.data['message_{}'.format(e)]
            nf_update_inputs.append(msg)
        # nf_update_inputs.append(nodes.data['nf'])
        nf_update_input = torch.cat(nf_update_inputs, dim=-1)
        updated_nf = self.node_update_func["{}".format(ntype)](nf_update_input)

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
