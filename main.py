import csv
from pathlib import Path
from typing import Tuple
import numpy as np
import IPython
import re


JOINTS = 20
GESTURES = 12


def to_gesture_id(gesture: str) -> int:
    regex = r"\d+"
    matches = re.findall(regex, gesture)
    return int(matches[0]) - 1


def to_microsecond(tick: int) -> int:
    return (tick*1000 + 49875/2)/49875


def load_datapoints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r') as f_data:
        reader = csv.reader(f_data, delimiter=' ')
        values = [[float(x) for x in row] for row in reader]
        values = np.array(values)
        timestamps = values[:, 0].reshape(-1, 1)
        X = values[:, 1:]
        return timestamps, X


def load_tagstream(path: Path) -> np.ndarray:
    with open(path, 'r') as f_tagstream:
        reader = csv.reader(f_tagstream, delimiter=';')
        tagstream = [row for row in reader]
        tagstream = np.array(tagstream)
        tagstream = tagstream[1:, :]
        tagstream[:, 1] = np.vectorize(to_gesture_id)(tagstream[:, 1])
        tagstream = np.asarray(tagstream, dtype=int)
        tagstream[:, 0] = np.vectorize(to_microsecond)(tagstream[:, 0])
        return tagstream


def tagstream_to_y(timestamps: np.ndarray, X: np.ndarray, tagstream: np.ndarray) -> np.ndarray:
    Y = np.zeros((X.shape[0], GESTURES))
    for row in tagstream:
        timestamp, gi = row
        frame = np.argmin(np.abs(timestamps[:, 0] - timestamp))
        Y[frame][gi] = 1
    return Y


def load_file(datapoints: Path, tagstream: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps, X = load_datapoints(datapoints)
    tagstream = load_tagstream(tagstream)
    Y = tagstream_to_y(timestamps, X, tagstream)
    return X, Y


def main():
    X, Y = load_file('MicrosoftGestureDataset-RC/data/P1_1_1_p06.csv',
                     'MicrosoftGestureDataset-RC/data/P1_1_1_p06.tagstream')
    IPython.embed()


if __name__ == '__main__':
    main()
