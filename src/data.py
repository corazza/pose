import IPython
import os
import csv
from pathlib import Path
from typing import Generator, Tuple
import re
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import const
import graph as ourgraph


def to_gesture_id(gesture: str) -> int:
    regex = r"\d+"
    matches = re.findall(regex, gesture)
    return int(matches[0]) - 1


def to_microsecond(tick: int) -> int:
    return (tick*1000 + 49875/2)/49875


def first_nonzero(X: np.ndarray) -> int:
    return np.argmax(np.any(X != 0, axis=1))


def load_datapoints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r') as f_data:
        reader = csv.reader(f_data, delimiter=' ')
        values = [[float(x) for x in row] for row in reader]
        values = np.array(values)
        timestamps = values[:, 0].reshape(-1, 1)
        X = values[:, 1:]
        nonzero_i = first_nonzero(X)
        X = X[nonzero_i:, :]
        timestamps = timestamps[nonzero_i:, :]
        keep_cols = np.ones(X.shape[1], dtype=bool)
        keep_cols[3::4] = False
        X = X[:, keep_cols]
        X = X.reshape(X.shape[0], -1, const.FEATURES)
        return timestamps.reshape((timestamps.shape[0])), X


def load_tagstream(path: Path) -> np.ndarray:
    with open(path, 'r') as f_tagstream:
        reader = csv.reader(f_tagstream, delimiter=';')
        tagstream = np.array([row for row in reader])
        tagstream = tagstream[1:, :]  # remove "XQPCTick;Tag"
        tagstream[:, 1] = np.vectorize(to_gesture_id)(
            tagstream[:, 1])  # e.g. G1 -> 0
        tagstream = np.asarray(tagstream, dtype=int)
        tagstream[:, 0] = np.vectorize(to_microsecond)(tagstream[:, 0])
        return tagstream


def tagstream_to_y(timestamps: np.ndarray, X: np.ndarray, tagstream: np.ndarray) -> np.ndarray:
    Y = np.zeros((X.shape[0], const.GESTURES))
    for row in tagstream:
        timestamp, gi = row
        frame = np.argmin(np.abs(timestamps[:] - timestamp))
        Y[frame][gi] = 1
    return Y


def extract_label(Y: np.ndarray) -> np.ndarray:
    return Y[(Y == 1).any(axis=1)][0]


def split(timestamps: np.ndarray, X: np.ndarray, Y: np.ndarray) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    row_indices = np.where(np.any(Y == 1, axis=1))[0]
    # distances = [abs(row_indices[i+1] - row_indices[i])
    #              for i in range(len(row_indices)-1)]
    # min_cover = min(distances)/2
    min_cover = int(const.WINDOW/2)
    Xs = []
    Ys = []
    Ts = []
    for center in row_indices:
        a = center - min_cover
        b = center + min_cover
        if a < 0 or b > X.shape[0]:
            continue
        Xs.append(X[a:b, :])
        Ys.append(Y[a:b, :])
        Ts.append(timestamps[a:b])
    return Ts, Xs, Ys


def load_file(datapoints: Path, tagstream: Path) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    timestamps, X = load_datapoints(datapoints)
    tagstream = load_tagstream(tagstream)
    Y = tagstream_to_y(timestamps, X, tagstream)
    Ts, Xs, Ys = split(timestamps, X, Y)
    # will always be the same label for a single file
    Ys = list(map(extract_label, Ys))
    for T, X, Y in zip(Ts, Xs, Ys):
        yield T, X, Y


def examples(dir: str) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    dir = Path(dir)
    for file_path in dir.iterdir():
        if file_path.is_file():
            if '.tagstream' in str(file_path):
                continue
            data_path = file_path
            tagstream_path = Path(str(file_path).replace('.csv', '.tagstream'))
            if not os.path.exists(tagstream_path):
                continue
            yield from load_file(data_path, tagstream_path)


def get_data():
    a = torch.tensor(ourgraph.get_A(const.WINDOW))
    edge_index = a.nonzero().t().contiguous().long()

    all = list(examples('MicrosoftGestureDataset-RC/data'))
    Ts = list(map(lambda triplet: torch.tensor(triplet[0]), all))
    Xs = list(map(lambda triplet: torch.tensor(triplet[1]), all))
    ys = list(map(lambda triplet: torch.tensor(triplet[2]), all))

    Xs = list(map(lambda x: x.reshape(
        (x.shape[0]*x.shape[1], x.shape[2])), Xs))

    ys = list(map(lambda y: torch.argmax(y), ys))

    data_list = []
    for X, y in zip(Xs, ys):
        data = Data(x=X.float(), y=y.long(), edge_index=edge_index.long())
        data_list.append(data)

    train_size = int(0.9 * len(data_list))
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data_list, [train_size, test_size])

    loader = DataLoader(
        train_dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        test_dataset, batch_size=const.BATCH_SIZE, shuffle=True)

    total_values = []
    for batch in loader:
        values = batch.x.view(batch.num_nodes, -1)
        total_values.append(values)
    total_values = torch.cat(total_values, dim=0)
    mean = torch.mean(total_values, dim=0)
    std = torch.std(total_values, dim=0)

    return loader, val_loader, mean, std, edge_index
