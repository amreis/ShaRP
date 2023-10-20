import argparse
import json
import os
import logging

from google.cloud import storage

CLASSES = {
    "ambulance",
    "bicycle",
    "bulldozer",
    "car",
    "firetruck",
    "motorbike",
    "pickup truck",
    "police car",
    "truck",
    "van",
}


def main():
    logger = logging.getLogger(__package__)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", "-f", help="Force re-downloading of data.", action="store_true")

    args = parser.parse_args()
    force = args.force

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket("quickdraw_dataset")

    index_to_class = {}

    for i, cl_name in enumerate(sorted(CLASSES)):
        fname = f"full/numpy_bitmap/{cl_name}.npy"
        if not force and os.path.exists(fname):
            continue
        blob = bucket.blob(fname)
        logger.info(f"downloading {cl_name}.npy")
        blob.download_to_filename(f"{cl_name}.npy")
        logger.info(f"downloaded {cl_name}.npy")
        index_to_class[i] = cl_name

    logger.info("saving key as json")
    with open("y_key.json", "wt") as f:
        json.dump(index_to_class, f)


if __name__ == "__main__":
    main()
