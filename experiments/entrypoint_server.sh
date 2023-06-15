#!/bin/bash

echo "nameserver 8.8.8.8" > /etc/resolv.conf
python3 ./experiment_server.py --verbose -o /opt/results/server
apt-get -y install zip
filename="results_server_$(date +"%Y-%m-%d-%H-%M-%S").zip"
zip -r "$filename" results
python3 upload_files.py "$filename" EdgedroidVol
sleep infinity


#    edgedroidserver
#    vnmo/edgedroid2:experiment-server-amd64v4
#    EDGEDROID_SERVER_BIND_ADDR=192.168.1.1,EDGEDROID_SERVER_BIND_PORT=50000,EDGEDROID_SERVER_TRACE=test,AUTH_SERVER=testbed.expeca.proj.kth.se,AUTH_PROJECT_NAME=edgedroid,AUTH_USERNAME=vishnu,AUTH_PASSWORD=Iniest@8
#    networks.1.interface=eno12409,networks.1.ip=192.168.1.1/24
#    e471d056-250e-4814-acad-5779595c81de

