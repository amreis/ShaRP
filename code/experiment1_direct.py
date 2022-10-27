#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image, ImageFont
from skimage import io
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP
from sklearn.manifold import Isomap

import ae
import metrics
import nnproj
import ssnp
import sharp

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None, X_test=None, X_inv_test=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    DSC = metrics.distance_consistency(X_2d, y)
    CC = metrics.cluster_size_consistency_r(X, y, X_2d)

    if X_inv is not None:
        MSE_train = metrics.metric_mse(X, X_inv)
    else:
        MSE_train = -99.0

    if X_inv_test is not None:
        assert X_test is not None, "if X_inv_test is provided, X_test must be too"
        MSE_test = metrics.metric_mse(X_test, X_inv_test)
    else:
        MSE_test = -99.0

    return T, C, R, S, N, DSC, CC, MSE_train, MSE_test


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment_1")

    parser.add_argument("--output-dir", type=str, default="results_direct")
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
    classes_mult["usps"] = 2

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset[dataset_name]

        sharp_gt = sharp.ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.1),
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
        X_inv_sharp_gt = sharp_gt.inverse_transform(X_sharp_gt)
        X_inv_sharp_gt_test = sharp_gt.inverse_transform(sharp_gt.transform(X_test))

        sharp_km = sharp.ShaRP(
            X.shape[1],
            n_clusters,
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.1),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = KMeans(n_clusters=n_clusters)
        y_km = C.fit_predict(X_train)
        sharp_km.fit(X_train, y_km, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_km = sharp_km.transform(X_train)
        X_inv_sharp_km = sharp_km.inverse_transform(X_sharp_km)
        X_inv_sharp_km_test = sharp_km.inverse_transform(sharp_km.transform(X_test))

        sharp_ag = sharp.ShaRP(
            X.shape[1],
            n_clusters,
            "diagonal_normal",
            variational_layer_kwargs=dict(kl_weight=0.1),
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        C = AgglomerativeClustering(n_clusters=n_clusters)
        y_ag = C.fit_predict(X_train)
        sharp_ag.fit(X_train, y_ag, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp_ag = sharp_ag.transform(X_train)
        X_inv_sharp_ag = sharp_ag.inverse_transform(X_sharp_ag)
        X_inv_sharp_ag_test = sharp_ag.inverse_transform(sharp_ag.transform(X_test))

        ssnpgt = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)
        X_inv_ssnpgt = ssnpgt.inverse_transform(X_ssnpgt)
        X_inv_ssnpgt_test = ssnpgt.inverse_transform(ssnpgt.transform(X_test))

        ssnpkm = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        C = KMeans(n_clusters=n_clusters)
        y_km = C.fit_predict(X_train)
        ssnpkm.fit(X_train, y_km)
        X_ssnpkm = ssnpkm.transform(X_train)
        X_inv_ssnpkm = ssnpkm.inverse_transform(X_ssnpkm)
        X_inv_ssnpkm_test = ssnpkm.inverse_transform(ssnpkm.transform(X_test))

        ssnpag = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        C = AgglomerativeClustering(n_clusters=n_clusters)
        y_ag = C.fit_predict(X_train)
        ssnpag.fit(X_train, y_ag)
        X_ssnpag = ssnpag.transform(X_train)
        X_inv_ssnpag = ssnpag.inverse_transform(X_ssnpag)
        X_inv_ssnpag_test = ssnpag.inverse_transform(ssnpag.transform(X_test))

        tsne = TSNE(n_jobs=4, random_state=420)
        X_tsne = tsne.fit_transform(X_train)

        ump = UMAP(random_state=420)
        X_umap = ump.fit_transform(X_train)

        X_isomap = Isomap().fit_transform(X_train)

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)
        X_inv_aep = aep.inverse_transform(X_aep)
        X_inv_aep_test = aep.inverse_transform(aep.transform(X_test))

        nnp = nnproj.NNProj(init=TSNE(n_jobs=4, random_state=420))
        nnp.fit(X_train)
        X_nnp = nnp.transform(X_train)

        D_sharp_gt = metrics.compute_distance_list(X_sharp_gt)
        D_sharp_km = metrics.compute_distance_list(X_sharp_km)
        D_sharp_ag = metrics.compute_distance_list(X_sharp_ag)
        D_ssnpgt = metrics.compute_distance_list(X_ssnpgt)
        D_ssnpkm = metrics.compute_distance_list(X_ssnpkm)
        D_ssnpag = metrics.compute_distance_list(X_ssnpag)
        D_tsne = metrics.compute_distance_list(X_tsne)
        D_umap = metrics.compute_distance_list(X_umap)
        D_isomap = metrics.compute_distance_list(X_isomap)
        D_aep = metrics.compute_distance_list(X_aep)
        D_nnp = metrics.compute_distance_list(X_nnp)

        results.append(
            (dataset_name, "ShaRP-GT")
            + compute_all_metrics(
                X_train,
                X_sharp_gt,
                D_high,
                D_sharp_gt,
                y_train,
                X_inv_sharp_gt,
                X_test,
                X_inv_sharp_gt_test,
            )
        )
        results.append(
            (dataset_name, "ShaRP-KMeans")
            + compute_all_metrics(
                X_train,
                X_sharp_km,
                D_high,
                D_sharp_km,
                y_train,
                X_inv_sharp_km,
                X_test,
                X_inv_sharp_km_test,
            )
        )
        results.append(
            (dataset_name, "ShaRP-AG")
            + compute_all_metrics(
                X_train,
                X_sharp_ag,
                D_high,
                D_sharp_ag,
                y_train,
                X_inv_sharp_ag,
                X_test,
                X_inv_sharp_ag_test,
            )
        )
        results.append(
            (dataset_name, "SSNP-GT")
            + compute_all_metrics(
                X_train,
                X_ssnpgt,
                D_high,
                D_ssnpgt,
                y_train,
                X_inv_ssnpgt,
                X_test,
                X_inv_ssnpgt_test,
            )
        )
        results.append(
            (dataset_name, "SSNP-KMeans")
            + compute_all_metrics(
                X_train,
                X_ssnpkm,
                D_high,
                D_ssnpkm,
                y_train,
                X_inv_ssnpkm,
                X_test,
                X_inv_ssnpkm_test,
            )
        )
        results.append(
            (dataset_name, "SSNP-AG")
            + compute_all_metrics(
                X_train,
                X_ssnpag,
                D_high,
                D_ssnpag,
                y_train,
                X_inv_ssnpag,
                X_test,
                X_inv_ssnpag_test,
            )
        )
        results.append(
            (dataset_name, "AE")
            + compute_all_metrics(
                X_train, X_aep, D_high, D_aep, y_train, X_inv_aep, X_test, X_inv_aep_test
            )
        )
        results.append(
            (dataset_name, "TSNE") + compute_all_metrics(X_train, X_tsne, D_high, D_tsne, y_train)
        )
        results.append(
            (dataset_name, "UMAP") + compute_all_metrics(X_train, X_umap, D_high, D_umap, y_train)
        )
        results.append(
            (dataset_name, "ISOMAP")
            + compute_all_metrics(X_train, X_isomap, D_high, D_isomap, y_train)
        )
        results.append(
            (dataset_name, "NNP") + compute_all_metrics(X_train, X_nnp, D_high, D_nnp, y_train)
        )

        for X_, label in zip(
            [
                X_sharp_gt,
                X_sharp_km,
                X_sharp_ag,
                X_ssnpgt,
                X_ssnpkm,
                X_ssnpag,
                X_umap,
                X_tsne,
                X_isomap,
                X_aep,
                X_nnp,
            ],
            [
                "ShaRP-GT",
                "ShaRP-KMeans",
                "ShaRP-AG",
                "SSNP-GT",
                "SSNP-KMeans",
                "SSNP-AG",
                "UMAP",
                "TSNE",
                "ISOMAP",
                "AE",
                "NNP",
            ],
        ):
            fname = os.path.join(output_dir, "{0}_{1}.png".format(dataset_name, label))
            print(fname)
            plot(X_, y_train, fname)

    df = pd.DataFrame(
        results,
        columns=[
            "dataset_name",
            "test_name",
            "T_train",
            "C_train",
            "R_train",
            "S_train",
            "N_train",
            "DSC_train",
            "CC_train",
            "MSE_train",
            "MSE_test",
        ],
    )

    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)

    # don't plot NNP
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = [
        "ShaRP-GT",
        "ShaRP-KMeans",
        "ShaRP-AG",
        "SSNP-KMeans",
        "SSNP-AG",
        "AE",
        "TSNE",
        "ISOMAP",
        "UMAP",
        "SSNP-GT",
    ]

    images = glob(output_dir + "/*.png")
    base = 2000

    for d in data_dirs:
        dataset_name = d
        to_paste = []

        for i, label in enumerate(pri_images):
            to_paste += [
                f
                for f in images
                if os.path.basename(f) == "{0}_{1}.png".format(dataset_name, label)
            ]

        img = np.zeros((base, base * len(pri_images), 3)).astype("uint8")

        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:, i * base : (i + 1) * base, :] = tmp[:, :, :3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + "/composite_full_{0}.png".format(dataset_name))

        for i, label in enumerate(pri_images):
            print(
                "/composite_full_{0}.png".format(dataset_name),
                "{0} {1}".format(dataset_name, label),
            )

    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = ["SSNP-KMeans", "SSNP-AG", "AE"]

    images = glob(output_dir + "/*.png")
    base = 2000

    for d in data_dirs:
        dataset_name = d
        to_paste = []

        for i, label in enumerate(pri_images):
            to_paste += [
                f
                for f in images
                if os.path.basename(f) == "{0}_{1}.png".format(dataset_name, label)
            ]

        img = np.zeros((base, base * 3, 3)).astype("uint8")

        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:, i * base : (i + 1) * base, :] = tmp[:, :, :3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + "/composite_{0}.png".format(dataset_name))

        for i, label in enumerate(pri_images):
            print(
                "/composite_{0}.png".format(dataset_name),
                "{0} {1}".format(dataset_name, label),
            )
