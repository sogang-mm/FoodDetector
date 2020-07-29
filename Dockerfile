FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update \
    && apt-get -y install python \
    python-pip \
    python-dev \
    git vim \
    openssh-server

RUN apt-get install -y --no-install-recommends tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .

RUN export PYTHONPATH=$PYTHONPATH:/workspace
RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace

