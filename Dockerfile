FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update \
    && apt-get -y install python \
    python-pip \
    python-dev \
    git\
    openssh-server

RUN pip install --upgrade pip
RUN pip install setuptools

WORKDIR /workspace
ADD . .

RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace

