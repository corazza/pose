import IPython
import torch
import sys
from torch_geometric.data import Data

import const
import graph as ourgraph
import model as ourmodel
import data as ourdata
import train
from process_video import video_to_tensor


def classify():
    torch.manual_seed(const.SEED)
    name = sys.argv[1]
    model, _mean_std = ourmodel.get_model(name)
    loader, val_loader, mean, std, edge_index = ourdata.get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # train.evaluate(-1, device, model, mean, std, val_loader)

    video = torch.tensor(video_to_tensor('iva_kick.mp4')).float()
    data = Data(x=video, edge_index=edge_index)
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.batch = batch
    # data = train.normalize(data, mean, std)
    data = data.to(device)

    model.eval()
    out = model(data.x, data.edge_index, data.batch)
    pred = torch.argmax(out, dim=-1)

    print(out)
    print(pred)
    IPython.embed()


def evaluate():
    torch.manual_seed(const.SEED)
    name = sys.argv[1]
    model, info = ourmodel.get_model(name)
    loader, val_loader, mean, std, edge_index = ourdata.get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    train.evaluate(-1, device, model, mean, std, val_loader)

    # ourmodel.save(name, model, mean, std, info['hidden'])


def main():
    classify()


if __name__ == '__main__':
    main()
