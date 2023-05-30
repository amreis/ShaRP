#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch, KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

import ae
import metrics
import ssnp
import sharp

if __name__ == "__main__":
    verbose = False
    results = []

    output_dir = "results_inverse"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = "../data/"
    data_dirs = ["mnist", "fashionmnist", "har", "reuters"]

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
    epochs_dataset["har"] = 10
    epochs_dataset["hatespeech"] = 20
    epochs_dataset["reuters"] = 10

    classes_mult = {}
    classes_mult["fashionmnist"] = 2
    classes_mult["mnist"] = 2
    classes_mult["har"] = 2
    classes_mult["hatespeech"] = 3
    classes_mult["reuters"] = 1

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

        train_size = 5000
        test_size = 1000

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            random_state=420,
            stratify=y,
        )
        print(X_train.shape, X_test.shape)

        epochs = epochs_dataset[dataset_name]

        sharp_model = sharp.ShaRP(
            X.shape[1],
            len(np.unique(y)),
            "laplace",
            variational_layer_kwargs=dict(prior_scale=0.1),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )
        sharp_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=64,
        )
        X_sharp = sharp_model.transform(X_train)
        X_inv_sharp = sharp_model.inverse_transform(X_sharp)
        X_sharp_test = sharp_model.transform(X_test)
        X_inv_sharp_test = sharp_model.inverse_transform(X_sharp_test)

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)
        X_inv_aep = aep.inverse_transform(X_aep)

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
        X_inv_ssnp = ssnpkm.inverse_transform(X_ssnpkm)

        X_aep_test = np.zeros((X_test.shape[0], 2))  # aep.transform(X_test)
        X_inv_aep_test = np.zeros_like(X_test)  # aep.inverse_transform(X_aep_test)

        X_ssnpkm_test = ssnpkm.transform(X_test)
        X_inv_ssnp_test = ssnpkm.inverse_transform(X_ssnpkm_test)

        results.append(
            (
                dataset_name,
                "SSNP-VAE",
                "train",
                metrics.metric_mse(X_train, X_inv_sharp),
            )
        )
        results.append(
            (
                dataset_name,
                "SSNP-KMeans",
                "train",
                metrics.metric_mse(X_train, X_inv_ssnp),
            )
        )
        results.append((dataset_name, "AE", "train", metrics.metric_mse(X_train, X_inv_aep)))

        results.append(
            (
                dataset_name,
                "SSNP-VAE",
                "test",
                metrics.metric_mse(X_test, X_inv_sharp_test),
            )
        )
        results.append(
            (
                dataset_name,
                "SSNP-KMeans",
                "test",
                metrics.metric_mse(X_test, X_inv_ssnp_test),
            )
        )
        results.append((dataset_name, "AE", "test", metrics.metric_mse(X_test, X_inv_aep_test)))

    df = pd.DataFrame(results, columns=["dataset_name", "test_name", "type", "mse"])

    df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)

    #######################################
    # create sample images
    dataset_name = "mnist"
    d = dataset_name

    X = np.load(os.path.join(data_dir, d, "X.npy"))
    y = np.load(os.path.join(data_dir, d, "y.npy"))

    n_classes = len(np.unique(y)) * classes_mult[dataset_name]
    n_samples = X.shape[0]

    train_size = min(int(n_samples * 0.9), 5000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=420, stratify=y
    )

    epochs = epochs_dataset[dataset_name]
    sharp_model = sharp.ShaRP(
        X.shape[1],
        len(np.unique(y)),
        "triangle",
        variational_layer_kwargs=dict(use_bias=True),
        var_leaky_relu_alpha=-0.0001,
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.5,
    )
    sharp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=verbose,
        batch_size=64,
    )
    X_sharp = sharp_model.transform(X_train)
    X_inv_sharp = sharp_model.inverse_transform(X_sharp)

    aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
    aep.fit(X_train)
    X_aep = aep.transform(X_train)
    X_inv_aep = aep.inverse_transform(X_aep)

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
    X_inv_ssnp = ssnpkm.inverse_transform(X_ssnpkm)

    base = 28
    img = np.zeros((base * 12, base * 10)).astype("float32")

    for i, x in enumerate(range(100, 110)):
        img[base * 0 : base * 1, i * base : (i + 1) * base] = X_train[x].reshape((28, 28))
        img[base * 1 : base * 2, i * base : (i + 1) * base] = X_inv_aep[x].reshape((28, 28))
        img[base * 2 : base * 3, i * base : (i + 1) * base] = X_inv_ssnp[x].reshape((28, 28))
        img[base * 3 : base * 4, i * base : (i + 1) * base] = X_inv_sharp[x].reshape((28, 28))

    for i, x in enumerate(range(200, 210)):
        img[base * 4 : base * 5, i * base : (i + 1) * base] = X_train[x].reshape((28, 28))
        img[base * 5 : base * 6, i * base : (i + 1) * base] = X_inv_aep[x].reshape((28, 28))
        img[base * 6 : base * 7, i * base : (i + 1) * base] = X_inv_ssnp[x].reshape((28, 28))
        img[base * 7 : base * 8, i * base : (i + 1) * base] = X_inv_sharp[x].reshape((28, 28))

    for i, x in enumerate(range(400, 410)):
        img[base * 8 : base * 9, i * base : (i + 1) * base] = X_train[x].reshape((28, 28))
        img[base * 9 : base * 10, i * base : (i + 1) * base] = X_inv_aep[x].reshape((28, 28))
        img[base * 10 : base * 11, i * base : (i + 1) * base] = X_inv_ssnp[x].reshape((28, 28))
        img[base * 11 : base * 12, i * base : (i + 1) * base] = X_inv_sharp[x].reshape((28, 28))

    print(y_train[100:110])
    print(y_train[200:210])
    print(y_train[400:410])

    io.imsave(os.path.join(output_dir, "inverse_mnist.png"), img)
