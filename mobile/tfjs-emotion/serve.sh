#!/bin/bash

cd app && python -m http.server &
ssh -R 80:localhost:8000 serveo.net