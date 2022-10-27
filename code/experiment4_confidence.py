#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP

import ae
import nnproj
import ssnp
import sharp
import var_metrics


def plot(X, y, figname=None):
    """Shamelessly stolen from Espadoto's SSNP code."""
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(20, 20))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=20)
        # ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig, ax


def plot_variance_maps(model, X_proj, base_fname: str, *, has_gradient: bool, has_log_var: bool):
    xmin, xmax = X_proj[:, 0].min(), X_proj[:, 0].max()
    ymin, ymax = X_proj[:, 1].min(), X_proj[:, 1].max()
    x_coords, y_coords = np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250)
    xx, yy = np.meshgrid(x_coords, y_coords)

    sample_points = np.c_[xx.ravel(), yy.ravel()]

    diffs = var_metrics.finite_differences(model, sample_points, eps=1e-2).numpy().sum(axis=1)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title(f"Finite Difference Gradient (min={diffs.min():.3g}, max={diffs.max():.3g})")
    ax.scatter(xx.ravel(), yy.ravel(), c=diffs, cmap="gray", marker=",")
    fig.savefig(f"{base_fname}_finitediffs.png")
    plt.close("all")
    if has_gradient:
        gradients = var_metrics.gradient_norm(model, sample_points).numpy()
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(f"Gradient Norm (min={gradients.min():.3g}, max={gradients.max():.3g})")
        ax.scatter(xx.ravel(), yy.ravel(), c=gradients, cmap="gray", marker=",")
        fig.savefig(f"{base_fname}_gradients.png")
        plt.close("all")

    if has_log_var:
        variances = np.exp(
            0.5 * model.log_var_embedding(model.inverse_transform(sample_points))
        ).sum(axis=1)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(
            f"Learned Variance Model (min={variances.min():.3g}, max={variances.max():.3g})"
        )
        ax.scatter(xx.ravel(), yy.ravel(), c=variances, cmap="gray", marker=",")
        fig.savefig(f"{base_fname}_variances.png")
        plt.close("all")


def plot_input_sensitivity(model, X, y, base_fname):
    classes = np.unique(y)
    for c in classes:
        inputs = X[y == c][:100]
        gradients = var_metrics.direct_proj_gradient(model, inputs)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title(f"Sensitivity for class {c}")
        ax.imshow(gradients.numpy().reshape((28, 28, 1)), cmap="gray")
        fig.savefig(f"{base_fname}_sensitivity_{c}.png")
    plt.close("all")


def make_plots(
    output_dir,
    base_fname,
    X_train,
    y_train,
    model,
    X_proj,
    *,
    has_gradient,
    has_log_var,
    plot_sensitivity,
):
    fname = os.path.join(output_dir, base_fname)
    plot(X_proj, y_train, f"{fname}.png")
    plot_variance_maps(model, X_proj, fname, has_gradient=has_gradient, has_log_var=has_log_var)
    if plot_sensitivity and has_gradient:
        plot_input_sensitivity(
            model.mu_model if hasattr(model, "mu_model") else model.fwd, X_train, y_train, fname
        )


if __name__ == "__main__":
    patience = 5
    epochs = 200

    min_delta = 0.05

    verbose = False
    results = []

    output_dir = "results_variance"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.join("..", "data")
    # skipping reuters because it does ginormous allocations. TODO try to fix?
    data_dirs = ["mnist", "fashionmnist", "har"]

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
        plot_sensitivity = "mnist" in dataset_name

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

        epochs = epochs_dataset[dataset_name]
        sharp = sharp.ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            "diagonal_normal",
            {
                "prior_loc": 0.0,
                "prior_scale": 1.0,
                "use_exact_kl": True,
                "kl_weight": 0.1,
                "kl_mu_weight": 0.01,
            },
            bottleneck_activation="linear",
        )
        sharp.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=64)
        X_sharp = sharp.transform(X_train)

        make_plots(
            output_dir,
            f"{dataset_name}_SSNP-VAE",
            X_train,
            y_train,
            sharp,
            X_sharp,
            has_gradient=True,
            has_log_var=True,
            plot_sensitivity=plot_sensitivity,
        )

        ssnpgt = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)

        make_plots(
            output_dir,
            f"{dataset_name}_SSNP-GT",
            X_train,
            y_train,
            ssnpgt,
            X_ssnpgt,
            has_gradient=True,
            has_log_var=False,
            plot_sensitivity=plot_sensitivity,
        )

        ssnpkm = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        ssnpkm.fit(X_train, y_km)
        X_ssnpkm = ssnpkm.transform(X_train)

        make_plots(
            output_dir,
            f"{dataset_name}_SSNP-KM",
            X_train,
            y_train,
            ssnpkm,
            X_ssnpkm,
            has_gradient=True,
            has_log_var=False,
            plot_sensitivity=plot_sensitivity,
        )

        ssnpag = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        )
        C = AgglomerativeClustering(n_clusters=n_classes)
        y_ag = C.fit_predict(X_train)
        ssnpag.fit(X_train, y_ag)
        X_ssnpag = ssnpag.transform(X_train)

        make_plots(
            output_dir,
            f"{dataset_name}_SSNP-AG",
            X_train,
            y_train,
            ssnpag,
            X_ssnpag,
            has_gradient=True,
            has_log_var=False,
            plot_sensitivity=plot_sensitivity,
        )

        ump = UMAP(random_state=420)
        X_umap = ump.fit_transform(X_train)

        # make_plots(
        #     output_dir,
        #     f"{dataset_name}_UMAP",
        #     X_train,
        #     y_train,
        #     ump,
        #     X_umap,
        #     has_gradient=False,
        #     has_log_var=False,
        #     plot_sensitivity=plot_sensitivity,
        # )
        fname = os.path.join(output_dir, f"{dataset_name}_UMAP")
        plot(X_umap, y_train, f"{fname}.png")

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)

        make_plots(
            output_dir,
            f"{dataset_name}_AE",
            X_train,
            y_train,
            aep,
            X_aep,
            has_gradient=True,
            has_log_var=False,
            plot_sensitivity=plot_sensitivity,
        )
