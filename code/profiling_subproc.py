import subprocess
import argparse
import os
import json

from datetime import datetime

import pandas as pd

from functools import partial


def run(alg: str, dataset: str, base_command: tuple[str, ...], timeout: float) -> float:
    try:
        command = base_command + ("--algorithm", alg, "--dataset", dataset)
        print(" ".join(command))
        cmmd = subprocess.run(
            command,
            timeout=timeout,
            capture_output=True,
        )
        lines = cmmd.stdout.splitlines()
        if lines:
            return float(lines[-1])
        else:
            return -1.0
    except (subprocess.TimeoutExpired, ValueError):
        # TimeoutExpired is for timeout
        # ValueError is for failed casting to float
        return -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=420)
    parser.add_argument("--output-dir", type=str, default="results_profiling_subproc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--to-subfolder", action="store_true", default=False)
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout in seconds for each training task"
    )

    args = parser.parse_args()

    run_datetime = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")

    seed = args.random_seed
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    train_size: int = args.train_size
    timeout: float = args.timeout

    if args.to_subfolder:
        output_dir = os.path.join(output_dir, run_datetime)

    config = {
        "verbose": False,
        "random_seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_size": train_size,
        "timeout": timeout,
    }
    print(f"Outputting results to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)
    dataset_names = os.listdir(os.path.join("..", "data"))
    results = []
    base_command = (
        "python",
        "profile_single_alg.py",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--train-size",
        str(train_size),
        "--batch-size",
        str(batch_size),
    )
    for dataset in dataset_names:
        run_with_alg = partial(run, dataset=dataset, base_command=base_command, timeout=timeout)
        sharp_time = run_with_alg("ShaRP")
        ssnp_time = run_with_alg("SSNP")
        isomap_time = run_with_alg("Isomap")
        tsne_time = run_with_alg("t-SNE")
        umap_time = run_with_alg("UMAP")
        aep_time = run_with_alg("AE")

        results.extend(
            [
                (dataset,) + technique_data
                for technique_data in [
                    ("sharp", sharp_time),
                    ("ssnp", ssnp_time),
                    ("tsne", tsne_time),
                    ("umap", umap_time),
                    ("isomap", isomap_time),
                    ("aep", aep_time),
                ]
            ]
        )
    pd.DataFrame(
        results,
        columns=["dataset", "algorithm", "time"],
    ).to_csv(os.path.join(output_dir, "profiling.csv"), header=True, index=False)


if __name__ == "__main__":
    main()
