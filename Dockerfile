FROM tensorflow/tensorflow:2.7.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
	apt-get install -y openjdk
