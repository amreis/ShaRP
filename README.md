# ShaRP

This repository contains the code and example scripts for ShaRP, the shape-regularized projection algorithm.

The main files for implementation are `sharp.py` and `sampling_layers.py`. Running `python sharp.py` executes a basic demo of the technique.


## Python Setup

You can run the code in at least two ways. If you use `pipenv`, simply run `pipenv install` while in the main directory, and a virtual environment will be set up with everything you need (recommended).

If you'd rather manage your environment yourself, we provide a `requirements.txt` file that you can use to install dependencies by running `pip install -r requirements.txt`.

## Obtain Datasets and Run Experiments
```bash
cd code
python get_data.py
python experiment1_direct.py
python experiment2_inverse.py
python experiment3_clustering.py
(...)
```
