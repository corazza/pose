import IPython
import sys
import torch
import torch.nn.functional as F

import const
import model as ourmodel
import data as ourdata
import graph as ourgraph


def normalize(batch, mean, std):
    X = batch.x
    X = (X - mean) / (std + 1e-6)
    batch.x = X
    return batch


def evaluate(epoch, device, model, mean, std, val_loader):
    model.eval()
    losses = []
    correct_preds = []
    for batch in val_loader:
        batch = normalize(batch, mean, std)
        batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = torch.argmax(out, dim=-1)
            val_loss = F.nll_loss(out, batch.y)
            losses.append(val_loss.item())
            num_correct = torch.sum(pred == batch.y).item()
            correct_preds.append(num_correct)
    avg_loss = sum(losses)/len(losses)
    accuracy = sum(correct_preds) / (len(correct_preds)*const.BATCH_SIZE)

    print(
        f'accuracy={accuracy}, avg_loss={avg_loss}, epoch={epoch+1}/{const.EPOCHS}')

    return accuracy


def main():
    torch.manual_seed(const.SEED)
    name = sys.argv[1]
    model, _mean_std = ourmodel.get_model(name)
    loader, val_loader, mean, std, edge_index = ourdata.get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    max_accuracy = evaluate(-1, device, model, mean, std, val_loader)
    print(f'running for {const.EPOCHS} epochs')
    for epoch in range(const.EPOCHS):
        model.train()
        for batch in loader:
            batch = normalize(batch, mean, std)
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
        accuracy = evaluate(epoch, device, model, mean, std, val_loader)
        if accuracy > max_accuracy:
            ourmodel.save(name, model, mean, std)
            max_accuracy = accuracy
    ourmodel.save(name, model, mean, std, last=True)


if __name__ == '__main__':
    main()
