# NN_P

Description:

This project is a custom implementation and replication of SchNet - graph continuous-filter neural network for modeling quantum interactions[1].
It uses PyTorch as its NN framework and ASE (Atomic Simulation Environment) for processing molecular information.
Python scripts for processing .npz files are provided so user can create their custom dataset.

Installation (tested with Python 3.12.3):

git clone https://github.com/nomouro/NN_Project.git

cd NN_Project

python -m venv venv_name

source ./venv_name/bin/activate

# You can choose between requirements.txt and requirements_working.txt , the latter was tested.

pip install -r requirements.txt

pip install -e .

Usage example:

The easiest way to train a model is to download existing MD17 dataset in the /DATA dir relative to cwd, configure (or use default) config.yaml file and type run in cli. You can do this in examples dir. It has the default structure and script that creates .db file from benzene MD17 dataset.

cd examples/N_1000_test/DATA

python3.12 db_creation.py

cd ..

run

Citation:

Schütt, Kristof, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in neural information processing systems 30 (2017).

https://doi.org/10.48550/arXiv.1706.08566

# License
This project is licensed under the terms of the MIT license.