#!/bin/bash
{ cd app && python -m http.server; } &
{ cd ssh -R 80:localhost:8000 serveo.net; } &
wait -n
pkill -P $$