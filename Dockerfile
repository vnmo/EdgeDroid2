FROM ubuntu:22.04 as base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y #  && apt-get upgrade -y
RUN apt-get install python3 python3-pip git build-essential cmake ffmpeg libsm6 libxext6 -y

COPY . /opt/edgedroid
WORKDIR /opt/edgedroid

FROM base as client
RUN pip install ".[load-client]"
RUN mkdir -p /edgedroid

WORKDIR /edgedroid
RUN edgedroid-tools --verbose prefetch-all-traces --yes
ENTRYPOINT ["edgedroid-client"]

FROM base as server
RUN pip install ".[load-server]"
RUN mkdir -p /edgedroid

WORKDIR /edgedroid
ENTRYPOINT ["edgedroid-server"]

