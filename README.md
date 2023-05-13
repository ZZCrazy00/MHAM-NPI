# MHANN-NPI

Predicting ncRNA-protein interactions based on
Multi-Head Attention neural networks

## Main requirements

python = 3.9

torch = 1.12.0

scikit-learn = 1.2.2

You can run the following code to generate the environment:

conda env create -f environment.yaml

## Usage

For the NPInter2 dataset, you need to run the sample.py file first to generate the sample.txt file, and then run main.py directly.

For RPI1807 and RPI488 datasets, just run main.py directly.