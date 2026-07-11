# Builds the ggml GPT-2 OpenAI-compatible API server.
# Bakes in MBZUAI/LaMini-GPT-124M (instruction-tuned GPT-2, ~124M params) at
# build time from ./models. See examples/gpt-2/convert-h5-to-ggml.py.
#
# NOTE: LaMini-GPT-124M is CC-BY-NC-4.0 licensed (non-commercial use only).
# For commercial use, swap in a permissively-licensed checkpoint instead.
#
#   docker build -t ggml-gpt2-server .
#   docker run -p 8080:8080 ggml-gpt2-server
#
# To swap in a different model without rebuilding, bind-mount it and point
# MODEL_PATH at it (set CHAT_TEMPLATE=raw for a non-instruction-tuned base
# GPT-2 checkpoint):
#   docker run -p 8080:8080 -v /path/to/model.bin:/models/other.bin \
#       -e MODEL_PATH=/models/other.bin -e CHAT_TEMPLATE=raw ggml-gpt2-server

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
COPY --from=builder --chown=ggml:ggml /src/models/lamini-gpt-124m/ggml-model.bin /models/ggml-model.bin

RUN ldconfig

ENV MODEL_PATH=/models/ggml-model.bin \
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
