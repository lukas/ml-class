#!/bin/bash
virtualenv --no-site-packages venv
. venv/bin/activate
pip install tensorflow_gpu==1.14.0 wandb pillow ipykernel
python -m ipykernel install --user --name=tf1.14