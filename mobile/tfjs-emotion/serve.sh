#!/bin/bash

trap 'kill $BGPID; exit' SIGINT
cd app && python -m http.server &
BGPID=$!
ssh -R 80:localhost:8000 serveo.net