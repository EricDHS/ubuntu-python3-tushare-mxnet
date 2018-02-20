FROM ubuntu:16.04
MAINTAINER haishandu "haishandu123@gmail.com"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev vim \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip install \
    pandas \
    lxml \
    requests\
    bs4
RUN pip install tushare \
    mxnet \
    gluon
