#!/bin/bash

tensorflowjs_converter --input_format keras --quantization_bytes 2 emotion.h5 app/models

echo "Model converted, run ./serve.sh"