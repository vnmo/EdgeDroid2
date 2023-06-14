FROM molguin/edgedroid2:latest as base
ENV DEBIAN_FRONTEND=noninteractive

COPY . /opt/edgedroid-experiments
WORKDIR /opt/edgedroid-experiments

RUN pip install -Ur ./requirements.txt

#    RUN pip install keystoneauth1 python-swiftclient
#    COPY upload_files.py /tmp/upload_files.py
#    COPY entrypoint_client.sh /tmp/entrypoint_client.sh
#    RUN chmod +x /tmp/entrypoint_client.sh
#    ENTRYPOINT ["/tmp/entrypoint_client.sh"]

FROM base as client
RUN pip install keystoneauth1 python-swiftclient
COPY upload_files.py /tmp/upload_files.py
COPY entrypoint_client.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
#ENTRYPOINT ["python3", "./experiment_client.py"]

FROM base as server
RUN pip install keystoneauth1 python-swiftclient
COPY upload_files.py /tmp/upload_files.py
COPY entrypoint_server.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
#ENTRYPOINT ["python3", "./experiment_server.py"]

