FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git \
    libtbb-dev libeigen3-dev libpcl-dev libyaml-cpp-dev libgflags-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
