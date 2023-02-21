import IPython
import torch
import sys

import const
import graph as ourgraph
import model as ourmodel
import data as ourdata
import train


def main():
    torch.manual_seed(const.SEED)
    name = sys.argv[1]
    model, _mean_std = ourmodel.get_model(name)
    loader, val_loader, mean, std = ourdata.get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train.evaluate(-1, device, model, mean, std, val_loader)

    IPython.embed()


if __name__ == '__main__':
    main()
