FROM continuumio/miniconda3

RUN mkdir /ml_drought
WORKDIR /ml_drought

COPY environment.ubuntu.cpu.yml .
RUN conda env create -f ./environment.ubuntu.cpu.yml

COPY src ./src
COPY tests ./tests
COPY *.ini ./
