FROM molguin/edgedroid2:latest as base
ENV DEBIAN_FRONTEND=noninteractive

COPY . /opt/edgedroid-experiments
WORKDIR /opt/edgedroid-experiments

RUN pip install -Ur ./requirements.txt

FROM base as client
ENTRYPOINT ["python3", "./experiment_client.py"]

FROM base as server
ENTRYPOINT ["python3", "./experiment_server.py"]

