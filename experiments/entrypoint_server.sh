#!/bin/bash

python3 ./experiment_server.py


#  $EDGEDROID_SERVER_BIND_ADDR       0.0.0.0
#  $EDGEDROID_SERVER_BIND_PORT       50000
#  $EDGEDROID_SERVER_TRACE           test

#   edgedroidserver
#   vnmo/edgedroid2:experiment-server-amd64
#   EDGEDROID_SERVER_BIND_ADDR=192.168.1.1,EDGEDROID_SERVER_BIND_PORT=50000,EDGEDROID_SERVER_TRACE=test
#   networks.1.interface=eno12409,networks.1.ip=192.168.1.1/24
#   e471d056-250e-4814-acad-5779595c81de

#  --rm -it
#  -p 5000:5000
#  --name server
#  -v ${PWD}/results/server:/opt/results/server:rw
#  --verbose
#  -o /opt/results/server

