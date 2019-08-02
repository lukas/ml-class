#!/bin/bash
virtualenv --no-site-packages venv
. venv/bin/activate
pip install tf-nightly-2.0-preview wandb pillow ipykernel
python -m ipykernel install --user --name=tf2.0