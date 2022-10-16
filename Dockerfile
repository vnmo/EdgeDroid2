#FROM ubuntu:22.04 as base
#ENV DEBIAN_FRONTEND=noninteractive
#
#RUN apt-get update -y #  && apt-get upgrade -y
#RUN apt-get install python3 python3-pip git build-essential cmake ffmpeg libsm6 libxext6 -y
FROM molguin/edgedroid2:dlib-base
ENV DEBIAN_FRONTEND=noninteractive

COPY . /opt/edgedroid
WORKDIR /opt/edgedroid

RUN pip install .
RUN edgedroid-fetch-all-traces
RUN mkdir -p /edgedroid

WORKDIR /edgedroid
