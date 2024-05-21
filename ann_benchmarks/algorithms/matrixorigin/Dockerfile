FROM ann-benchmarks
FROM golang:1.21.5-bookworm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get install -y --no-install-recommends build-essential make

RUN git clone https://github.com/matrixorigin/matrixone /tmp/matrixone

# Build and install matrixone
RUN cd /tmp/matrixone && \
    make clean && \
    make install
