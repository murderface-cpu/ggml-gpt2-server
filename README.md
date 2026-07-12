# ggml-gpt2-server

A small, self-hostable, OpenAI-API-compatible chat server. It's a [ggml](https://github.com/ggml-org/ggml)
GPT-2 architecture model behind an HTTP API you can point any OpenAI client
at (`base_url` + `api_key`), with streaming, concurrent request handling,
and one-command HTTPS deployment via Docker + Caddy.

Runs on CPU. No GPU required. Model choice is a runtime config setting, not
a rebuild, so you can trade off size/speed/quality for your VPS without
touching the image.

## What's in here

- `examples/gpt-2/server.cpp`: the server. Endpoints are `/v1/completions`,
  `/v1/chat/completions` (both with SSE streaming support), `/v1/models`,
  and `/health`. Model weights are loaded once and shared read-only; each
  in-flight request gets its own KV-cache "slot" from a small pool, so
  multiple requests are handled concurrently instead of queuing behind
  one another.
- `scripts/get_model.py`: downloads and converts any of the supported
  models by name.
- `examples/gpt-2/convert-h5-to-ggml.py`: converts a Hugging Face
  `GPT2LMHeadModel` checkpoint to the ggml binary format this server loads.
- `Dockerfile`, `docker-compose.yml`, `Caddyfile`: runs behind Caddy for
  automatic Let's Encrypt HTTPS. The model is mounted at runtime, not baked
  into the image, so switching models doesn't need a rebuild.
- `deploy.md`: step-by-step VPS deployment guide.
- `examples/gpt-2/tests/test_server.py`: black-box test suite against a
  running server (health, both endpoints, streaming, auth, concurrency,
  error handling), stdlib-only.
- `src/`, `include/`, `cmake/`: the underlying [ggml](https://github.com/ggml-org/ggml)
  tensor library this is built on (MIT licensed; see `LICENSE` and `AUTHORS`).

## Models

Run `python scripts/get_model.py --list` for the current set. As of writing:

| Name              | Size  | Instruction-tuned | Notes                                                        |
|-------------------|-------|--------------------|---------------------------------------------------------------|
| `lamini-124m`     | 124M  | yes                | Default. Fast, low memory, good for a small/cheap VPS.        |
| `lamini-774m`     | 774M  | yes                | Noticeably more reliable answers. Needs more RAM and is slower per token. See [Switching models](#switching-models) for the resource trade-off. |
| `gpt2-117m-base`  | 117M  | no                 | Original OpenAI GPT-2, permissively licensed, but rambles instead of answering.|
| `gpt2-345m-base`  | 345M  | no                 | Same, larger.                                                  |

The `lamini-*` models are [LaMini-LM](https://github.com/mbzuai-nlp/LaMini-LM)
checkpoints: GPT-2 fine-tuned on 2.58M distilled instructions, so they
actually follow instructions instead of free-associating like base GPT-2.
`lamini-774m` is the LaMini-LM paper's own pick for the best overall
size/performance trade-off in the series.

**License note:** the `lamini-*` models are CC-BY-NC-4.0, non-commercial
use only. The `gpt2-*-base` models are the original MIT-licensed OpenAI
GPT-2 weights, usable commercially, but they aren't instruction-tuned
(pair them with `CHAT_TEMPLATE=raw`).

## Quickstart (Docker, recommended)

```bash
git clone <this-repo-url>
cd ggml-gpt2-server

python scripts/get_model.py lamini-124m   # or lamini-774m, etc

docker compose up -d --build
```

See [`deploy.md`](deploy.md) for the full VPS walkthrough (DNS, firewall,
`.env` config, sizing, troubleshooting).

## Quickstart (build locally)

```bash
python scripts/get_model.py lamini-124m

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gpt-2-server

./build/bin/gpt-2-server -m models/lamini-124m/ggml-model.bin --port 8080
```

## Switching models

Models aren't baked into the Docker image. Fetch whichever ones you want
locally (they land in `models/<name>/ggml-model.bin`, gitignored), then
point the server at one:

```bash
python scripts/get_model.py lamini-774m
```

**Docker Compose**: set `MODEL_PATH` and `CHAT_TEMPLATE` in `.env` (see
`scripts/get_model.py`'s output for the exact values to use), then:

```bash
docker compose up -d
```

No rebuild needed; `./models` is bind-mounted into the container.

**Local build**: pass `-m` / `--chat-template` directly, or set the
`MODEL_PATH` / `CHAT_TEMPLATE` environment variables.

Resource cost scales with model size. Roughly, per concurrent request slot
at the default `CTX_SIZE=1024`:

| Model         | Weights (fixed) | Per slot | Per-token speed (relative) |
|---------------|------------------|----------|------------------------------|
| `lamini-124m` | ~315MB           | ~72MB    | baseline                     |
| `lamini-774m` | ~1.6GB           | ~360MB   | ~2x slower per token          |

See [Sizing in deploy.md](deploy.md#sizing) for concrete VPS size recommendations.

## API usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":30}'

# streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a short story"}],"max_tokens":100,"stream":true}'
```

Works with any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused-or-your-API_KEY")
resp = client.chat.completions.create(
    model="gpt-2",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Configuration

Environment variables (see `Dockerfile` for defaults):

| Variable           | Meaning                                                             |
|--------------------|----------------------------------------------------------------------|
| `MODEL_PATH`       | path to the `.bin` ggml model file                                   |
| `HOST` / `PORT`    | bind address (default `0.0.0.0:8080`)                                 |
| `CTX_SIZE`         | context window (default `1024`, matches all models listed above)     |
| `SLOTS`            | concurrent requests in flight (default: auto-sized from CPU count)   |
| `THREADS`          | CPU threads per in-flight request (default: auto-sized)              |
| `API_KEY`          | if set, requires `Authorization: Bearer <key>` (except `/health`)    |
| `CHAT_TEMPLATE`    | `alpaca` (default, for instruction-tuned models) or `raw` (plain transcript, for base GPT-2) |
| `QUEUE_TIMEOUT_MS` | how long a request waits for a free slot before `503`                |

## Testing

```bash
python examples/gpt-2/tests/test_server.py --base-url http://localhost:8080
```

## License

The ggml library itself is MIT licensed, see `LICENSE` and `AUTHORS`.
Bundled models each have their own license, see [Models](#models) above.
