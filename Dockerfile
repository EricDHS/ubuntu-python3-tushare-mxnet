FROM ubuntu:16.04
MAINTAINER haishandu "haishandu123@gmail.com"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev vim locales \
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

RUN echo 'set encoding=utf-8' >> /usr/share/vim/vimrc
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /tushare
