#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
from dataclasses import dataclass
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image, ImageFont
from skimage import io
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import metrics
import sharp

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class ExperimentConfig:
    layer: str
    layer_kwargs: dict

    def __init__(self, layer: str, **kwargs) -> None:
        self.layer = layer
        self.layer_kwargs = kwargs

    def describe(self) -> str:
        all_args = {"layer": self.layer} | self.layer_kwargs
        return ".".join(f"{k}:{v}" for k, v in all_args.items())


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    DSC = metrics.distance_consistency(X_2d, y)
    CC = 0.5 + 0.5 * metrics.cluster_size_consistency_r(X, y, X_2d)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0

    return T, C, R, S, N, DSC, CC, MSE


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
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=100)
        ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment_10")

    parser.add_argument("--output-dir", type=str, default="results_shapes")
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
    print(f"Outputting data to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = "../data"
    data_dirs = args.datasets  # ["mnist", "fashionmnist", "har", "reuters", "usps"]
    print(data_dirs)

    experiment_configs = [
        ExperimentConfig("diagonal_normal", kl_weight=0.01),
        ExperimentConfig("diagonal_normal", kl_weight=0.1),
        ExperimentConfig("generalized_normal", power=5),
        ExperimentConfig("generalized_normal", power=15),
        ExperimentConfig("triangle", use_bias=True),
        ExperimentConfig("triangle", use_bias=False),
    ]

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10 * 2
    epochs_dataset["mnist"] = 10 * 2
    epochs_dataset["har"] = 10 * 2
    epochs_dataset["reuters"] = 10 * 2
    epochs_dataset["usps"] = 10 * 2

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

        n_clusters = len(np.unique(y)) * classes_mult[dataset_name]
        n_samples = X.shape[0]

        train_size = min(int(n_samples * 0.9), 5000)

        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset[dataset_name]

        for exp in experiment_configs:
            print(f"{exp!r}")
            subdir = os.path.join(output_dir, exp.describe())
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            variational_layer = exp.layer
            variational_layer_kwargs = exp.layer_kwargs
            sharp_gt = sharp.ShaRP(
                X.shape[1],
                len(np.unique(y_train)),
                variational_layer,
                variational_layer_kwargs=variational_layer_kwargs,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.5,
            )
            sharp_gt.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=verbose,
                batch_size=64,
            )
            X_sharp_gt = sharp_gt.transform(X_train)
            X_sharp_gt_inv = sharp_gt.inverse_transform(X_sharp_gt)

            sharp_km = sharp.ShaRP(
                X.shape[1],
                n_clusters,
                variational_layer,
                variational_layer_kwargs=variational_layer_kwargs,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.5,
            )
            C = KMeans(n_clusters=n_clusters)
            y_km = C.fit_predict(X_train)
            sharp_km.fit(X_train, y_km, epochs=epochs, verbose=verbose, batch_size=64)
            X_sharp_km = sharp_km.transform(X_train)
            X_sharp_km_inv = sharp_km.inverse_transform(X_sharp_km)

            sharp_ag = sharp.ShaRP(
                X.shape[1],
                n_clusters,
                variational_layer,
                variational_layer_kwargs=variational_layer_kwargs,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.5,
            )
            C = AgglomerativeClustering(n_clusters=n_clusters)
            y_ag = C.fit_predict(X_train)
            sharp_ag.fit(X_train, y_ag, epochs=epochs, verbose=verbose, batch_size=64)
            X_sharp_ag = sharp_ag.transform(X_train)
            X_sharp_ag_inv = sharp_ag.inverse_transform(X_sharp_ag)

            D_sharp_gt = metrics.compute_distance_list(X_sharp_gt)
            D_sharp_km = metrics.compute_distance_list(X_sharp_km)
            D_sharp_ag = metrics.compute_distance_list(X_sharp_ag)

            results.append(
                (dataset_name, "ShaRP-GT", exp.describe())
                + compute_all_metrics(
                    X_train, X_sharp_gt, D_high, D_sharp_gt, y_train, X_sharp_gt_inv
                )
            )
            results.append(
                (dataset_name, "ShaRP-KMeans", exp.describe())
                + compute_all_metrics(
                    X_train, X_sharp_km, D_high, D_sharp_km, y_train, X_sharp_km_inv
                )
            )
            results.append(
                (dataset_name, "ShaRP-AG", exp.describe())
                + compute_all_metrics(
                    X_train, X_sharp_ag, D_high, D_sharp_ag, y_train, X_sharp_ag_inv
                )
            )
            for X_, label in zip(
                [
                    X_sharp_gt,
                    X_sharp_km,
                    X_sharp_ag,
                ],
                [
                    "ShaRP-GT",
                    "ShaRP-KMeans",
                    "ShaRP-AG",
                ],
            ):
                fname = os.path.join(subdir, "{0}_{1}.png".format(dataset_name, label))
                print(fname)
                plot(X_, y_train, fname)

    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "test_name",
            "exp",
            "T_train",
            "C_train",
            "R_train",
            "S_train",
            "N_train",
            "DSC_train",
            "CC_train",
            "MSE_train",
        ],
    )

    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)
