import argparse
import contextlib
import os
import gc
from time import perf_counter
from sys import stderr
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

import sharp
import ssnp
import ae
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP
from sklearn.model_selection import train_test_split
import numpy as np


class PerfTimer(contextlib.AbstractContextManager):
    def __init__(self) -> None:
        self._timer = perf_counter
        self._start = 0.0
        self._elapsed = None

    @property
    def elapsed(self):
        assert self._elapsed is not None, "PerfTimer must first be used to measure some code"
        return self._elapsed

    def __enter__(self) -> "PerfTimer":
        self._start = self._timer()
        return self

    def __exit__(self, *exc) -> bool:
        self._elapsed = self._timer() - self._start
        return True


def make_sharp(
    dim_input: int, n_classes: int, *, var_layer: str = "diagonal_normal", var_layer_kwargs=dict()
):
    return sharp.ShaRP(
        dim_input,
        n_classes,
        variational_layer=var_layer,
        variational_layer_kwargs=var_layer_kwargs,
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
    )


def make_ssnp(epochs: int, verbose: bool):
    return ssnp.SSNP(
        epochs=epochs, verbose=verbose, patience=0, opt="adam", bottleneck_activation="linear"
    )


def read_and_setup_data(
    dataset: str, train_size: int, random_seed: int = 420
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(os.path.join("..", "data", dataset, "X.npy"))
    y = np.load(os.path.join("..", "data", dataset, "y.npy"))
    print(X.shape, y.shape, file=stderr)

    train_size_prop = 0.9
    if train_size > train_size_prop * X.shape[0]:
        # Upsample
        print("upsampling", file=stderr)
        upsample_factor = np.ceil(train_size / X.shape[0])
        X = np.repeat(X, upsample_factor, axis=0)
        y = np.repeat(y, upsample_factor, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=min(train_size, int(train_size_prop * X.shape[0])),
        stratify=y,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test  # type: ignore


def main():
    gc.disable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ShaRP", "SSNP", "t-SNE", "UMAP", "AE", "Isomap"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot", action="store_true", default=False)

    args = parser.parse_args()
    dataset = args.dataset
    algorithm = args.algorithm
    epochs = args.epochs
    seed = args.seed
    train_size = args.train_size
    batch_size = args.batch_size

    X_train, _, y_train, _ = read_and_setup_data(dataset, train_size, seed)

    if algorithm == "ShaRP":
        sharp = make_sharp(
            X_train.shape[1],
            len(np.unique(y_train)),
            var_layer="diagonal_normal",
            var_layer_kwargs=dict(kl_weight=0.1),
        )
        with PerfTimer() as t:
            sharp.fit(X_train, y_train, epochs=epochs, verbose=False, batch_size=batch_size)
            x = sharp.transform(X_train)
        print(t.elapsed)
    elif algorithm == "SSNP":
        ssnp = make_ssnp(epochs=epochs, verbose=False)
        with PerfTimer() as t:
            ssnp.fit(X_train, y_train)
            x = ssnp.transform(X_train)
        print(t.elapsed)
    elif algorithm == "t-SNE":
        with PerfTimer() as t:
            x = TSNE(n_jobs=4, random_state=seed).fit_transform(X_train)
        print(t.elapsed)
    elif algorithm == "UMAP":
        with PerfTimer() as t:
            x = UMAP(random_state=420).fit_transform(X_train)
        print(t.elapsed)
    elif algorithm == "AE":
        with PerfTimer() as t:
            aep = ae.AutoencoderProjection(epochs=epochs, verbose=False)
            aep.fit(X_train)
            x = aep.transform(X_train)
        print(t.elapsed)
    elif algorithm == "Isomap":
        with PerfTimer() as t:
            x = Isomap().fit_transform(X_train)
        print(t.elapsed)

    if args.plot:
        plt.scatter(x[:, 0], x[:, 1], c=y_train)
        plt.show()


if __name__ == "__main__":
    main()
