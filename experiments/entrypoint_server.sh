#!/bin/bash

python3 ./experiment_server.py

#  --rm -it
#  -p 5000:5000
#  --name server
#  -v ${PWD}/results/server:/opt/results/server:rw
#  --verbose
#  -o /opt/results/server