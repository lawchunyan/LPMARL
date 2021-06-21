import numpy as np
import dgl
import torch

AG_NODE = 0
EN_NODE = 1

EN2AG_EDGE = 0
AG2AG_EDGE = 1
SELF_EDGE = 2


def make_graph(n_ag, n_en):
    # ag_state = np.array(ag_state)
    # en_state = np.array(en_state / 2)
    #
    # pad_len = ag_state.shape[1] - en_state.shape[1]
    # new_en_state = np.pad(en_state, ((0, 0), (0, pad_len)))
    #
    # n_ag = ag_state.shape[0]
    # n_en = en_state.shape[0]
    # full_state = np.concatenate([ag_state, new_en_state], axis=0)
    #
    # full_state[:n_ag] = ag_state

    # add nodes
    g = dgl.DGLGraph()
    g.add_nodes(n_ag + n_en)
    # g.ndata['node_feature'] = torch.Tensor(full_state)

    # node_type
    ntype = [AG_NODE for _ in range(n_ag)] + [EN_NODE for _ in range(n_en)]
    g.ndata['node_type'] = torch.Tensor(ntype)

    # add edges
    ag_range = range(n_ag)
    en_range = range(n_ag, n_ag + n_en)

    en2ag_from = [i for _ in ag_range for i in en_range]
    en2ag_to = [i for i in ag_range for _ in en_range]
    g.add_edges(en2ag_from, en2ag_to)

    ag2ag_from = [i for i in ag_range for _ in ag_range]
    ag2ag_to = [i for _ in ag_range for i in ag_range]
    g.add_edges(ag2ag_from, ag2ag_to)

    self_from = [i for i in ag_range] + [i for i in en_range]
    self_to = [i for i in ag_range] + [i for i in en_range]
    g.add_edges(self_from, self_to)

    edata = [EN2AG_EDGE for _ in en2ag_to] + [AG2AG_EDGE for _ in ag2ag_to] + [SELF_EDGE for _ in range(n_ag + n_en)]
    g.edata['edge_type'] = torch.Tensor(edata)

    return g