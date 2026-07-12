# llm-server

A small, self-hostable, OpenAI-API-compatible chat server: [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
own server, deployed with Docker + Caddy for automatic HTTPS. Point any
OpenAI client at it (`base_url` + `api_key`), get streaming, real
concurrency (continuous batching), and a genuinely long context window.

Runs on CPU. No GPU required. No build step: `docker compose up -d` pulls
the official llama.cpp server image directly.

## Why llama.cpp

This started as a hand-rolled GPT-2 server built directly on the `ggml`
tensor library. GPT-2 turned out to be a dead end for anything beyond a
toy: it's hard-capped at 1024 tokens of context (a fixed-size learned
position-embedding table baked into the weights, not a config setting),
and getting real long-context support would have meant reimplementing a
chunk of what llama.cpp already does well: RoPE, grouped-query attention,
GGUF loading, continuous batching. So this now just deploys llama.cpp's
own server instead of maintaining a custom one.

## What's in here

- `docker-compose.yml`, `Caddyfile`: `llm-server` (the official
  `ghcr.io/ggml-org/llama.cpp:server` image) behind Caddy for automatic
  Let's Encrypt HTTPS. The model is mounted at runtime, not baked into an
  image, so switching models is a config change and restart, not a rebuild.
- `scripts/get_model.py`: downloads a supported GGUF model by name.
- `deploy.md`: step-by-step VPS deployment guide.
- `tests/test_server.py`: black-box test suite against a running server
  (health, both endpoints, streaming, auth, concurrency, error handling),
  stdlib-only.

## Models

Run `python scripts/get_model.py --list` for the current set. As of writing:

| Name             | Size | Context | License    | Notes                                    |
|------------------|------|---------|------------|-------------------------------------------|
| `qwen2.5-1.5b`   | 1.5B | 32K     | Apache-2.0 | Default. Strong quality for its size.     |
| `qwen2.5-0.5b`   | 0.5B | 32K     | Apache-2.0 | Lighter/faster, for a smaller VPS.        |

Both are official GGUF quantizations (Q4_K_M) from Qwen's own Hugging Face
org, instruction-tuned, RoPE-based (real long context, unlike the GPT-2
models this project used to run).

## Quickstart (Docker, recommended)

```bash
git clone <this-repo-url>
cd llm-server

pip install huggingface_hub
python scripts/get_model.py qwen2.5-1.5b

docker compose up -d
```

See [`deploy.md`](deploy.md) for the full VPS walkthrough (DNS, firewall,
`.env` config, sizing, troubleshooting).

## Switching models

```bash
python scripts/get_model.py qwen2.5-0.5b
```

Set `MODEL_PATH` in `.env` to the printed path, then `docker compose up -d`.
No rebuild needed; `./models` is bind-mounted into the container.

llama.cpp can also pull a model directly with no separate download step at
all, via `-hf <repo>:<quant>` instead of `-m <path>` — see the image's
`--help`. `scripts/get_model.py` exists for the explicit
download-then-mount workflow `docker-compose.yml` uses, which avoids
depending on llama.cpp's HF-cache directory persisting across restarts.

Resource cost scales with context size and model size. Roughly, per
concurrent request slot:

| Model          | Weights (fixed) | Per slot @ CTX_SIZE=8192 | Per slot @ 32768 |
|----------------|------------------|----------------------------|---------------------|
| `qwen2.5-0.5b` | ~500MB           | ~96MB                     | ~384MB               |
| `qwen2.5-1.5b` | ~1.1GB           | ~224MB                    | ~896MB               |

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
    model="qwen2.5",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Configuration

Environment variables, set in `.env` (see `docker-compose.yml` for defaults):

| Variable    | Meaning                                                          |
|-------------|--------------------------------------------------------------------|
| `MODEL_PATH`| path to the `.gguf` model file, inside the container (under `/models`) |
| `CTX_SIZE`  | context window (default `8192`; the default model supports up to `32768`) |
| `SLOTS`     | concurrent requests in flight (`-np`)                              |
| `THREADS`   | CPU threads per in-flight request                                  |
| `API_KEY`   | if set, requires `Authorization: Bearer <key>` on requests          |
| `DOMAIN`    | your domain, for Caddy's automatic HTTPS cert                       |

## Testing

```bash
python tests/test_server.py --base-url http://localhost:8080
```

## License

MIT, see `LICENSE`, for what's authored in this repo (deployment config,
scripts, docs). Actual inference happens in the `ghcr.io/ggml-org/llama.cpp`
image, which this repo depends on but doesn't vendor or redistribute.
Bundled models each have their own license, see [Models](#models) above.
