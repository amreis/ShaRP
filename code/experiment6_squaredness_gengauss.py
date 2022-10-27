#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import callbacks

import metrics
import sharp

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        cmap = ListedColormap(
            [
                "#aee39a",
                "#9e37d0",
                "#7cee4d",
                "#713d83",
                "#1be19f",
                "#fb2076",
                "#458612",
                "#e89ff0",
                "#115d52",
                "#f79302",
            ]
        )

    else:
        cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(40, 40))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=100, alpha=0.5)
        ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax


def main():
    parser = argparse.ArgumentParser(description="experiment_6")
    parser.add_argument("--output-dir", type=str, default="results_squaredness_gengauss")
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "fashionmnist", "har", "reuters", "usps"]
    )

    args = parser.parse_args()
    patience = 5
    epochs = 200

    min_delta = 0.05

    verbose = False
    results = []

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = "../data"
    data_dirs = args.datasets

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10 * 5
    epochs_dataset["mnist"] = 10 * 5
    epochs_dataset["har"] = 10 * 5
    epochs_dataset["reuters"] = 10 * 5
    epochs_dataset["usps"] = 10 * 5

    classes_mult = {}
    classes_mult["fashionmnist"] = 2
    classes_mult["mnist"] = 2
    classes_mult["har"] = 2
    classes_mult["reuters"] = 1
    classes_mult["usps"] = 1

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
        for model_config in (
            dict(power=2.0),
            dict(power=3.0),
            dict(power=10.0),
            dict(power=15.0),
            dict(power=30.0),
            dict(power=100.0),
        ):
            sharp = sharp.ShaRP(
                X.shape[1],
                len(np.unique(y_train)),
                "generalized_normal",
                model_config,
                bottleneck_activation="linear",
            )
            sharp.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=verbose,
                batch_size=64,
                callbacks=[callbacks.TerminateOnNaN()],
            )
            X_sharp = sharp.transform(X_train)
            D_sharp = metrics.compute_distance_list(X_sharp)
            results.append(
                (dataset_name, f'{model_config["power"]:.3g}')
                + compute_all_metrics(X_train, X_sharp, D_high, D_sharp, y_train)
            )
            fname = os.path.join(
                output_dir,
                (f"{dataset_name}" f"_pow={model_config['power']:.3g}.png"),
            )
            print(fname)
            plot(X_sharp, y_train, fname)
    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "power",
            "T_train",
            "C_train",
            "R_train",
            "S_train",
            "N_train",
            "MSE_train",
        ],
    )

    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)


if __name__ == "__main__":
    main()
