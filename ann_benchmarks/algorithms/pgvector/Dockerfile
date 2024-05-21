FROM ann-benchmarks

FROM postgres:16.2

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

FROM pgvector/pgvector:pg16

ENV POSTGRES_PASSWORD=ann \
    POSTGRES_USER=ann \
    POSTGRES_DB=ann
EXPOSE 5432