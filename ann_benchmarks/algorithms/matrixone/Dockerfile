# Start from a base image that includes ann-benchmarks dependencies
FROM ann-benchmarks

# Install necessary packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    cmake \
    wget \
    g++ \
    python3 \
    python3-setuptools \
    python3-pip \
    libblas-dev \
    liblapack-dev \
    mysql-client \
    tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install a specific version of Go
ENV GO_VERSION=1.21.1
RUN wget https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go$GO_VERSION.linux-amd64.tar.gz && \
    rm go$GO_VERSION.linux-amd64.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"

# Clone the MatrixOne repository
RUN git clone https://github.com/matrixorigin/matrixone /tmp/matrixone

# Build MatrixOne
WORKDIR /tmp/matrixone
RUN go mod tidy && \
    make clean && \
    make && \
    ./mo-service -debug-http :9876 -launch ./etc/launch/launch.toml >out.log 2>err.log

# Create the 'ann' database using MySQL client
RUN mysql -h 127.0.0.1 -P 6001 -u dump -p111 -e "CREATE DATABASE ann;"

# Install Python packages
RUN pip3 install --no-cache-dir sqlalchemy numpy
