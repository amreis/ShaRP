import argparse
import pathlib
from itertools import starmap

import matplotlib.pyplot as plt
import metrics
import numpy as np
import plotly.express as px
import radial
import tensorflow as tf
from keras import callbacks
from sharp import ShaRP, compute_all_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def gen_plotly_plots(X, X_proj, y, y_key, base_path=pathlib.Path(".")):
    cmap = plt.colormaps.get_cmap("tab10")

    def to_css(r: int, g: int, b: int, _a: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    css_colors = list(starmap(to_css, [cmap(i, bytes=True) for i in range(10)]))

    fig = px.scatter_3d(
        x=X_proj[:, 0],
        y=X_proj[:, 1],
        z=X_proj[:, 2],
        color=y,
        color_discrete_sequence=css_colors,
        hover_data={
            "x": X_proj[:, 0],
            "y": X_proj[:, 1],
            "z": X_proj[:, 2],
            "color": y,
            "class": [y_key[lbl] for lbl in y],
        },
    )
    fig.write_html(base_path / "plotly_sharp.html")

    from MulticoreTSNE import MulticoreTSNE as TSNE

    X_tsne_2d = TSNE(n_jobs=8).fit_transform(X)
    fig = px.scatter(
        x=X_tsne_2d[:, 0],
        y=X_tsne_2d[:, 1],
        color=y,
        color_discrete_sequence=css_colors,
        hover_data={
            "x": X_tsne_2d[:, 0],
            "y": X_tsne_2d[:, 1],
            "color": y,
            "class": [y_key[lbl] for lbl in y],
        },
    )
    fig.write_html(base_path / "plotly_tsne_2d.html")
    X_tsne_3d = TSNE(n_components=3, n_jobs=8).fit_transform(X)
    fig = px.scatter_3d(
        x=X_tsne_3d[:, 0],
        y=X_tsne_3d[:, 1],
        z=X_tsne_3d[:, 2],
        color=y,
        color_discrete_sequence=css_colors,
        hover_data={
            "x": X_tsne_3d[:, 0],
            "y": X_tsne_3d[:, 1],
            "z": X_tsne_3d[:, 2],
            "color": y,
            "class": [y_key[lbl] for lbl in y],
        },
    )
    fig.write_html(base_path / "plotly_tsne_3d.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results_spherical")
    parser.add_argument("--datasets", nargs="+", default=["quickdraw2"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=40)

    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()
    data_dir = pathlib.Path("../data")
    data_dirs: list[str] = args.datasets
    print(data_dirs)

    epochs = args.epochs

    for d in data_dirs:
        dataset_name = d

        (output_dir / d).mkdir(exist_ok=True)

        X = np.load(data_dir / d / "X.npy")
        y = np.load(data_dir / d / "y.npy")
        if (fname := data_dir / d / "y_key.json").exists():
            import json

            with open(fname) as f:
                y_key = json.load(f)
            y_key = {int(k): v for k, v in y_key.items()}
        else:
            y_key = {i: str(i) for i in np.unique(y)}
        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=min(5_000, int(0.9 * X.shape[0])), random_state=420, stratify=y
        )
        label_bin = LabelBinarizer().fit(y_train)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, (label_bin.transform(y_train), X_train))
        ).shuffle(buffer_size=1024)
        val_dataset = train_dataset.take(500).batch(32)
        train_dataset = train_dataset.skip(500).take(30000).batch(64)

        sharp_gt = ShaRP(
            X.shape[1],
            len(np.unique(y_train)),
            variational_layer="spherical",
            variational_layer_kwargs={},
            latent_dim=3,
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        )

        sharp_gt.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=args.verbose,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=1e-3,
                    patience=20,
                    restore_best_weights=True,
                    mode="min",
                ),
                callbacks.TerminateOnNaN(),
            ],
        )

        X_i = sharp_gt.transform(X_train)

        gen_plotly_plots(
            X_train,
            X_i,
            y_train,
            y_key,
            output_dir / d,
        )


if __name__ == "__main__":
    main()
