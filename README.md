<h1 align="center">Emotion Classification with MobileNet Transfer Learning</h1>

<p align="center">
  <a href="emtionRecognition.py"><img src="https://img.shields.io/badge/Project-emtionRecognition.py-555555.svg" alt="Project entry file"></a>
  <a href="https://scholar.google.com/citations?user=bvyKhaEAAAAJ&hl=en"><img src="https://img.shields.io/badge/Publications-Google_Scholar-4285F4.svg" alt="Google Scholar"></a>
  <a href="https://www.kaggle.com/nizamuddinmaitlo"><img src="https://img.shields.io/badge/Profile-Kaggle-20BEFF.svg" alt="Kaggle profile"></a>
</p>

<p align="center"><b>Repository maintained by Nizamuddin Maitlo</b></p>

<p align="center">Folder-based grayscale emotion classification using an ImageNet-initialized MobileNet backbone.</p>

## 🔥 Overview

This repository fine-tunes a compact classifier on top of a frozen MobileNet feature extractor. Grayscale emotion images are resized, expanded to three channels, augmented during training, and evaluated on a separate test directory.

## ✨ Features

- Grayscale-to-RGB conversion for MobileNet input.
- ImageDataGenerator augmentation and validation split.
- ImageNet-initialized MobileNet feature extractor.
- Training curves, test accuracy, and sample visualization.

## 🧪 Method and protocol

- Training images must follow a class-folder hierarchy under `train`.
- Test images must use the same class folders under `test`.
- Twenty percent of the training folder is reserved for validation.
- Edit `train_dir` and `test_dir` in `emtionRecognition.py` before running.

## 📁 Repository contents

| File | Purpose |
|---|---|
| `emtionRecognition.py` | Data loading, MobileNet model, training, evaluation, and plots |

## 🛠️ Setup

Install the dependencies:

~~~bash
python -m pip install tensorflow numpy matplotlib
~~~

## 📦 Data and inputs

| Resource | Purpose | Availability |
|---|---|---|
| Folder-based grayscale emotion dataset | Training, validation, and held-out test images | User-provided dataset |

The image dataset is not stored in this repository, and no public dataset URL is currently associated with the script.

## 🚀 Running the project

Update the dataset paths in the script, then run:

~~~bash
python emtionRecognition.py
~~~

## ♻️ Reproducibility

- Record the Python and library versions used for each run.
- Keep preprocessing, splits, thresholds, and random seeds fixed when comparing results.
- Do not commit private input data, generated model weights, or machine-specific paths.
- Revalidate results when the dataset, sensor, operating environment, or dependency versions change.

## 📚 Publications

No paper-specific DOI is currently associated with this repository. This section is intentionally kept separate from related publications to avoid implying a publication-to-code relationship that has not been established.



## ⚠️ Scope and limitations

The current script contains machine-specific absolute paths and freezes the MobileNet backbone. Results depend on dataset balance, label quality, subject separation, and whether identities or near-duplicate frames cross the splits.

## 📄 License

No standalone license file is currently included in this repository.

## 🤝 Acknowledgements

This project uses open-source Python libraries and the data or inputs described above. We thank the original dataset, framework, and software contributors.
