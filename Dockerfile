FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Kiru Park (park@acin.tuwien.ac.at)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3.5-dev python3-pip python3-tk vim git libgtk2.0-dev 
pip3 install --upgrade -r requirements.txt

