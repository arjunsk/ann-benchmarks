FROM golang:1.22.3-bookworm as builder

# goproxy
ARG GOPROXY="https://proxy.golang.org,direct"
RUN go env -w GOPROXY=${GOPROXY}

RUN mkdir -p /go/src/github.com/matrixorigin/matrixone

WORKDIR /go/src/github.com/matrixorigin/matrixone

# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN set -xe; git init; \
    git remote add origin https://github.com/matrixorigin/matrixone.git; \
    git fetch origin main; \
    git checkout -b main origin/main;

RUN make build

# Start from a base image that includes ann-benchmarks dependencies
FROM ann-benchmarks

# Install necessary packages
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends mysql-client; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*;

COPY --from=builder /go/src/github.com/matrixorigin/matrixone/mo-service mo-service
COPY --from=builder /go/src/github.com/matrixorigin/matrixone/etc etc

RUN  ./mo-service -debug-http=:9876 -daemon -launch ./etc/launch/launch.toml >out.log 2>err.log; \
     while true; do if ! mysql -h 127.0.0.1 -P 6001 -u dump -p111 -e "CREATE DATABASE ann;"; then echo "mo init not finished...";sleep 1;else break; fi;done

# Install Python packages
RUN pip3 install --no-cache-dir sqlalchemy numpy