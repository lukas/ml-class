#!/usr/bin/env python

from subprocess import Popen
import time

server = Popen(["cd ./app && python -m http.server"], shell=True,
               stdin=None, stdout=None, stderr=None, close_fds=True)
tunnel = Popen(["ssh -R 80:localhost:8000 serveo.net"], shell=True,
               stdin=None, stdout=None, stderr=None, close_fds=True)

while True:
    time.sleep(1)
