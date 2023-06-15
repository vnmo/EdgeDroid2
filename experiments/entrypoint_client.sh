#!/bin/bash

echo "nameserver 8.8.8.8" > /etc/resolv.conf
python3 ./experiment_client.py --verbose -o /opt/results/client
apt-get -y install zip
filename="results_client_$(date +"%Y-%m-%d-%H-%M-%S").zip"
zip -r "$filename" results
python3 upload_files.py "$filename" EdgedroidVol
sleep infinity


#    edgedroidclient
#    vnmo/edgedroid2:experiment-client-amd64v4
#    EDGEDROID_CLIENT_HOST=192.168.1.1,EDGEDROID_CLIENT_PORT=50000,EDGEDROID_CLIENT_TRACE=test,EDGEDROID_CLIENT_EXPERIMENT_ID=empirical-high-adaptive-power-empirical,AUTH_SERVER=testbed.expeca.proj.kth.se,AUTH_PROJECT_NAME=edgedroid,AUTH_USERNAME=vishnu,AUTH_PASSWORD=Iniest@8
#    networks.1.interface=eno12419,networks.1.ip=192.168.1.2/24
#    e471d056-250e-4814-acad-5779595c81de

#  Options:
#    --truncate INTEGER              Truncate the specified task trace to a given
#                                    number of steps. Note that the server needs
#                                    to be configured with the same value for the
#                                    emulation to work.
#    -o, --output-dir DIRECTORY
#    -v, --verbose                   Enable verbose logging.
#    --connect-timeout-seconds FLOAT
#                                    Time in seconds before the initial
#                                    connection establishment times out.
#                                    [default: 5.0]
#    --max-connection-attempts INTEGER
#                                    Maximum connection retries, set to a 0 or a
#                                    negative value for infinite retries.
#                                    [default: 5]
