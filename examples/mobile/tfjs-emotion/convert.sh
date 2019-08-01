#!/bin/bash
virtualenv --no-site-packages venv
. venv/bin/activate
pip install tensorflowjs

tensorflowjs_converter --input_format keras --quantization_bytes 2 emotion.h5 app/models

deactivate
echo "Model converted, run python serve.py"