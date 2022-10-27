import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import callbacks

import metrics
import sharp


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    DSC = metrics.distance_consistency(X_2d, y)
    CC = metrics.cluster_size_consistency_r(X, y, X_2d)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0

    return T, C, R, S, N, DSC, CC, MSE


class MonitorQualityMetrics(callbacks.Callback):
    def __init__(self, X, y):
        super().__init__()
        self._X = np.copy(X)
        self._y = np.copy(y)

        self._pdistances = metrics.compute_distance_list(self._X)

        self.metrics = dict()

    def _compute_batch_metrics(self):
        X_proj = self.model.transform(self._X)
        lowd_pdistances = metrics.compute_distance_list(X_proj)

        return compute_all_metrics(self._X, X_proj, self._pdistances, lowd_pdistances, self._y)

    def on_epoch_end(self, epoch: int, logs=None):
        self.metrics[epoch + 1] = self._compute_batch_metrics()

    def on_train_begin(self, logs=None):
        self.metrics[0] = self._compute_batch_metrics()


def main():
    import matplotlib

    matplotlib.rcParams.update({"font.size": 22})
    parser = argparse.ArgumentParser("profiling")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--random-seed", "-s", type=int, default=420)
    parser.add_argument("--output-dir", "-o", type=str, default="results_qualityovertime")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--datasets", nargs="*", default=os.listdir(os.path.join("..", "data")))

    args = parser.parse_args()
    verbose: bool = args.verbose
    seed: int = args.random_seed
    output_dir: str = args.output_dir
    epochs: int = args.epochs
    batch_size: int = args.batch_size

    print(f"Outputting results to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_names = args.datasets

    results = dict()
    for dataset in dataset_names:
        X = np.load(os.path.join("..", "data", dataset, "X.npy"))
        y = np.load(os.path.join("..", "data", dataset, "y.npy"))
        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=5000, stratify=y, random_state=seed
        )
        sharp_gt = sharp.ShaRP(
            X_train.shape[1],
            len(np.unique(y_train)),
            "diagonal_normal",
            dict(kl_weight=0.1),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        monitor_quality = MonitorQualityMetrics(X_train, y_train)
        sharp_gt.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[monitor_quality],
        )

        results[dataset] = [(k,) + v for k, v in monitor_quality.metrics.items()]

    df = pd.DataFrame(
        itertools.chain.from_iterable(
            [[(d_name,) + m for m in ms] for d_name, ms in results.items()]
        ),
        columns=["dataset_name", "epoch", "T", "C", "R", "S", "N", "DSC", "CC", "MSE"],
    )
    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)

    translate_metric = {
        "T": "Trustworthiness",
        "C": "Continuity",
        "R": "Sheperd Diagram Corr.",
        "S": "Stress",
        "N": "Neighborhood Hit",
        "D": "Distance Consistency",
    }
    translate_dataset = {
        "mnist": "MNIST",
        "fashionmnist": "FashionMNIST",
        "reuters": "Reuters",
        "har": "HAR",
        "usps": "USPS",
    }
    linestyles_per_ds = {
        "mnist": "solid",
        "fashionmnist": "dotted",
        "reuters": "dashed",
        "har": "dashdot",
        "usps": (5, (10, 3)),
    }
    fig, axes = plt.subplot_mosaic(
        """
        TCR
        SND
        """,
        figsize=(20, 20),
    )
    for i, ax_name in enumerate(("T", "C", "R", "S", "N", "D")):
        ax: Axes = axes[ax_name]
        ax.set_title(translate_metric[ax_name])
        for d in dataset_names:
            d_results = results[d]
            ax.plot([res[1 + i] for res in d_results], label=translate_dataset[d])
        if ax_name != "R":
            ax.set_ylim(0, 1)
        ax.legend()

    plt.savefig(os.path.join(output_dir, "quality.png"))
    plt.close("all")

    for i, ax_name in enumerate(("T", "C", "R", "S", "N", "D")):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(translate_metric[ax_name])
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Value")
        for d in dataset_names:
            d_results = results[d]
            ax.plot(
                [res[1 + i] for res in d_results],
                label=translate_dataset[d],
                lw=3,
                linestyle=linestyles_per_ds[d],
            )
        if ax_name != "R":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.savefig(os.path.join(output_dir, f"{translate_metric[ax_name]}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
