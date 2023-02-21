import IPython
import os
from pathlib import Path
import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import aggr
from torch.nn import Linear, ReLU, Dropout, LogSoftmax
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import Sequential, GCNConv

import const


def get_paths(name: str, last: bool):
    if last:
        model_path = Path(f'/content/pose/models/{name}_last.pt')
    else:
        model_path = Path(f'/content/pose/models/{name}.pt')
    norm_path = Path(f'/content/pose/models/{name}_norm.pt')
    return model_path, norm_path


def get_model(name: str, hidden: int = None):
    model_path, norm_path = get_paths(name, False)
    if hidden == None:
        print(model_path, norm_path)
        mean_std = torch.load(norm_path)
        hidden = mean_std['hidden']
    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(const.FEATURES, hidden), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(hidden, hidden), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge("cat", hidden, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * hidden, const.GESTURES),
        LogSoftmax(dim=1)
    ])
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        mean_std = torch.load(norm_path)
        print(f'loaded model from {model_path.resolve()}')
    else:
        print('created new model')
        mean_std = None
    return model, mean_std


def save(name, model, mean, std, hidden, last=False):
    model_path, norm_path = get_paths(name, last)
    info = {'mean': mean, 'std': std, 'hidden': hidden}
    torch.save(info, norm_path)
    torch.save(model.state_dict(), model_path)
    print(f'saved model to {model_path.resolve()}')
