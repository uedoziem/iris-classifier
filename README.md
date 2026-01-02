\# Iris Classifier (Decision Tree)



\## Overview

This project trains a \*\*Decision Tree\*\* classifier on the classic \*\*Iris\*\* dataset using \*\*scikit-learn\*\*.

It provides:

\- A reproducible CLI training script (`src/train.py`)

\- Model evaluation using \*\*accuracy\*\*

\- A saved \*\*confusion matrix\*\* figure in `outputs/confusion\_matrix.png`



\## Quick start (Windows / Git Bash)

```bash

\# 1) Clone the repo

git clone https://github.com/<YOUR\_USERNAME>/iris-classifier.git

cd iris-classifier



\# 2) Create and activate a virtual environment

python -m venv venv

source venv/Scripts/activate



\# 3) Install dependencies

python -m pip install --upgrade pip

pip install -r requirements.txt



\# 4) Run training (prints accuracy + saves confusion matrix image)

python src/train.py --test-size 0.2 --random-state 42



