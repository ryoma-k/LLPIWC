FROM nvidia/cuda:12.0.1-base-ubuntu22.04

USER root

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y
RUN apt -y install build-essential gcc g++ make libtool texinfo dpkg-dev pkg-config git vim wget curl

RUN apt remove --purge python3.8
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -yq --no-install-recommends python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH /root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

WORKDIR /cache
RUN wget https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-linux_x86_64.whl
RUN wget https://download.pytorch.org/whl/cu117/torchvision-0.14.0%2Bcu117-cp310-cp310-linux_x86_64.whl

WORKDIR /work
COPY pyproject.toml poetry.lock ./
RUN apt install python3.10-distutils -y
RUN python3 -m pip uninstall -y distro-info
RUN poetry install --no-root
RUN ln -s /usr/bin/python3 /usr/bin/python
