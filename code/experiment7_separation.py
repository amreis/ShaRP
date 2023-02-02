import itertools
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import metrics
import sharp


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0

    return T, C, R, S, N, MSE


def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(20, 20))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=20)
        ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax


if __name__ == "__main__":
    verbose = False
    results = []

    output_dir = "results_separation"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.join("..", "data")
    data_dirs = ["mnist", "fashionmnist", "har", "reuters"]

    epochs_dataset = {
        "fashionmnist": 10 * 2,
        "mnist": 10 * 2,
        "har": 10 * 2,
        "reuters": 10 * 2,
    }

    classes_mult = {
        "fashionmnist": 2,
        "mnist": 2,
        "har": 2,
        "reuters": 1,
    }

    for d in data_dirs:
        dataset_name = d

        X = np.load(os.path.join(data_dir, d, "X.npy"))
        y = np.load(os.path.join(data_dir, d, "y.npy"))

        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        n_classes = len(np.unique(y)) * classes_mult[dataset_name]
        n_samples = X.shape[0]

        train_size = min(int(n_samples * 0.9), 5000)
        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset[dataset_name]
        for layer, prior_scale in itertools.product(
            ["centered_diagonal_normal", "diagonal_normal", "laplace", "gumbel"], [1.0, 0.05, 0.01]
        ):
            sharp_model = sharp.ShaRP(
                X.shape[1],
                len(np.unique(y_train)),
                layer,
                {
                    "prior_loc": 0.0,
                    "prior_scale": prior_scale,
                    "kl_weight": 0.1,
                },
                bottleneck_activation="linear",
            )
            sharp_model.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
            X_proj = sharp_model.transform(X_train)
            D_low = metrics.compute_distance_list(X_proj)
            results.append(
                (dataset_name, layer, prior_scale)
                + compute_all_metrics(X_train, X_proj, D_high, D_low, y_train)
            )
            fname = os.path.join(
                output_dir,
                "{0}_{1}_sigma={2}.png".format(dataset_name, layer, prior_scale),
            )
            print(fname)
            plot(X_proj, y_train, fname)

    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "variational_layer",
            "prior_scale",
            "T_train",
            "C_train",
            "R_train",
            "S_train",
            "N_train",
            "MSE_train",
        ],
    )
    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)
