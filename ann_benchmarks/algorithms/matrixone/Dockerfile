FROM ann-benchmarks

RUN git clone https://github.com/matrixorigin/matrixone /tmp/matrixone

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    mysql-client \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Dynamically set OPTFLAGS based on the architecture
RUN ARCH=$(uname -m) && \
    cd /tmp/matrixone && \
    make && \
    ./mo-service -debug-http :9876 -launch ./etc/launch/launch.toml >out.log  2>err.log;

# Install MySQL server and client
RUN apt-get install -y mysql-client


RUN mysql -h 127.0.0.1 -P 6001 -u dump -p111 -e "CREATE DATABASE ann;"

RUN pip3 install --no-cache-dir sqlalchemy numpy
