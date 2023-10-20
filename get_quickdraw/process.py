import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def main():
    np.random.seed(420)
    with open("y_key.json") as key_file:
        y_key = json.load(key_file)
    Xs, ys = [], []
    for class_number, class_name in y_key.items():
        class_number = int(class_number)
        np_file = np.load(f"{class_name}.npy")
        np_file = np_file[np.random.choice(np_file.shape[0], size=10000)]

        Xs.append(np_file)
        ys.append(np.repeat([class_number], np_file.shape[0]))

    X = minmax_scale(np.concatenate(Xs, axis=0).astype(np.float32))
    y = np.concatenate(ys, axis=0).astype(np.int32)

    X, _ignore, y, _ignore = train_test_split(
        X, y, train_size=60000, stratify=y, random_state=420, shuffle=True
    )

    np.save("X.npy", X)
    np.save("y.npy", y)


if __name__ == "__main__":
    main()
