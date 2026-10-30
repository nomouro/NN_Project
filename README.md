# NN_P

Description:

This project is a custom replication and implementation of SchNet - graph continuous-filter neural network for modeling quantum interactions[1].
It uses torch library as its NN engine and ASE for processing molecular information.
Python scripts for processing .xyz and .npz files are provided so user can create their custom dataset.

Installation (tested with Python 3.12.3):

git clone https://github.com/....

cd NN_Project

python -m venv venv_name

source ./venv_name/bin/activate

# You can choose between requirements.txt and requirements_working.txt , the latter was tested.

pip install requirements.txt

pip install -e .


Usage example:

The easiest way to train a model is to download existing MD17 dataset in the /DATA dir relative to cwd, configure (or use default) config.yaml file and type run in cli. You can do this in examples dir. It has the default structure and script that creates .db file from benzene MD17 dataset.



