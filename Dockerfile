FROM tensorflow/tensorflow:1.8.0-gpu-py3
MAINTAINER Kiru Park (park@acin.tuwien.ac.at)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3.5-dev python3-pip python3-tk vim git libgtk2.0-dev 
WORKDIR /Pix2Pose
COPY . .
RUN pip3 install --upgrade -r requirements.txt

