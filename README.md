# ShaRP

This repository contains the code and example scripts for ShaRP, the shape-regularized projection algorithm.

The main files for implementation are `sharp.py` and `sampling_layers.py`. Running `python sharp.py` executes a basic demo of the technique.

## Citing this work

This work appeared at [EuroVA 2023](https://www.eurova.org/eurova-2023), a workshop co-located with [EuroVIS 2023](https://www.eurovis.org/) and has been published in the EuroGraphics Digital Library.

Machado, A., Telea, A., & Behrisch, M. (2023). [ShaRP: Shape-Regularized Multidimensional Projections](https://diglib.eg.org/handle/10.2312/eurova20231088). In EuroVis Workshop on Visual Analytics (EuroVA). The Eurographics Association.

You can use the following BibTeX entry to cite it:


```bibtex
@inproceedings {10.2312:eurova.20231088,
booktitle = {EuroVis Workshop on Visual Analytics (EuroVA)},
editor = {Angelini, Marco and El-Assady, Mennatallah},
title = {{ShaRP: Shape-Regularized Multidimensional Projections}},
author = {Machado, Alister and Telea, Alexandru and Behrisch, Michael},
year = {2023},
publisher = {The Eurographics Association},
ISSN = {2664-4487},
ISBN = {978-3-03868-222-6},
DOI = {10.2312/eurova.20231088}
}
```

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


----

This algorithm takes heavy inspiration from [SSNP](https://research.rug.nl/en/publications/self-supervised-dimensionality-reduction-with-neural-networks-and), and some of the code here is copied and adapted from [its repository](https://github.com/mespadoto/ssnp).
