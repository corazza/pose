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


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(const.FEATURES, 32)
        self.conv2 = GCNConv(32, 16)
        self.sum_aggr = aggr.MeanAggregation()
        self.fc1 = torch.nn.Linear(16, const.GESTURES)
        # self.fc2 = torch.nn.Linear(16, const.GESTURES)
        # mean and std added so they get saved in the model
        self.mean = None
        self.std = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.sum_aggr(x, ptr=data.ptr)
        x = self.fc1(x)
        # x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_paths(name: str, last: bool):
    if last:
        model_path = Path(f'models/{name}_last.pt')
    else:
        model_path = Path(f'models/{name}.pt')
    norm_path = Path(f'models/{name}_norm.pt')
    return model_path, norm_path


def get_model(name: str):
    model_path, norm_path = get_paths(name, False)
    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(const.FEATURES, const.N_HIDDEN), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(const.N_HIDDEN, const.N_HIDDEN), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge("cat", const.N_HIDDEN, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * const.N_HIDDEN, const.GESTURES),
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


def save(name, model, mean, std, last=False):
    model_path, norm_path = get_paths(name, last)
    mean_std = {'mean': mean, 'std': std}
    torch.save(mean_std, norm_path)
    torch.save(model.state_dict(), model_path)
    print(f'saved model to {model_path.resolve()}')
