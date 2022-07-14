FROM tensorflow/tensorflow:2.7.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
	apt-get install -y openjdk-11-jdk \
                       python3.8-venv
                       
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64