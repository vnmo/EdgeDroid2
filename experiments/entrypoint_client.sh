#!/bin/bash

python3 ./experiment_client.py -o /opt/results/client
python3 upload_files.py /tmp/results/irtt_data.json EdgedroidVol

#  $EDGEDROID_CLIENT_HOST              <serverIP>
#  $EDGEDROID_CLIENT_PORT              50000
#  $EDGEDROID_CLIENT_TRACE             test
#  $EDGEDROID_CLIENT_EXPERIMENT_ID     empirical-high-adaptive-power-empirical


#   edgedroidclient
#   vnmo/edgedroid2:experiment-client-amd64
#   EDGEDROID_CLIENT_HOST=192.168.1.1,EDGEDROID_CLIENT_PORT=50000,EDGEDROID_CLIENT_TRACE=test,EDGEDROID_CLIENT_EXPERIMENT_ID=empirical-high-adaptive-power-empirical
#   networks.1.interface=eno12419,networks.1.ip=192.168.1.2/24
#   e471d056-250e-4814-acad-5779595c81de

#  --rm -it
#  --name=client
#  -v ${PWD}/results/client:/opt/results/client:rw
#  --verbose
#  -o /opt/results/client



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
