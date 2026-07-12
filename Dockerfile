# Builds the ggml GPT-2 OpenAI-compatible API server.
#
# The model is NOT baked into the image. Mount your local ./models directory
# and pick which one to load via MODEL_PATH, so switching models is just a
# config change and a restart, not a rebuild:
#
#   python scripts/get_model.py lamini-124m     # or lamini-774m, etc
#   docker build -t ggml-gpt2-server .
#   docker run -p 8080:8080 -v "$(pwd)/models:/models:ro" \
#       -e MODEL_PATH=/models/lamini-124m/ggml-model.bin \
#       -e CHAT_TEMPLATE=alpaca \
#       ggml-gpt2-server
#
# See scripts/get_model.py --list for available models, and README.md for
# more on switching models (docker-compose.yml already wires this up via
# MODEL_PATH/CHAT_TEMPLATE in .env).

FROM debian:bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# GGML_NATIVE=OFF: don't tune for the build host's CPU, since the image may be
# built on a different machine than the one it eventually runs on.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
    && cmake --build build --config Release --target gpt-2-server -j "$(nproc)"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 ggml

COPY --from=builder /src/build/bin/gpt-2-server /usr/local/bin/gpt-2-server
COPY --from=builder /src/build/src/*.so* /usr/local/lib/

RUN ldconfig

ENV MODEL_PATH=/models/lamini-124m/ggml-model.bin \
    CHAT_TEMPLATE=alpaca \
    HOST=0.0.0.0 \
    PORT=8080 \
    CTX_SIZE=1024 \
    API_KEY=
# SLOTS / THREADS are intentionally left unset here so the server can
# auto-size them from the container's visible CPU count; override both
# explicitly (e.g. via docker-compose.yml) for predictable sizing.

EXPOSE 8080

USER ggml
ENTRYPOINT ["gpt-2-server"]
