#!/bin/bash
virtualenv --no-site-packages venv
. venv/bin/activate
pip install tensorflow-gpu==2.0.0b1 wandb pillow

python cnn.py

deactivate