# ggml-gpt2-server

A small, self-hostable, OpenAI-API-compatible chat server. It's a [ggml](https://github.com/ggml-org/ggml)
GPT-2 architecture model behind an HTTP API you can point any OpenAI client
at (`base_url` + `api_key`), with streaming, concurrent request handling,
and one-command HTTPS deployment via Docker + Caddy.

Runs on CPU. No GPU required. The default model is small enough to run
comfortably on a $5-10/mo VPS.

## What's in here

- `examples/gpt-2/server.cpp`: the server. Endpoints are `/v1/completions`,
  `/v1/chat/completions` (both with SSE streaming support), `/v1/models`,
  and `/health`. Model weights are loaded once and shared read-only; each
  in-flight request gets its own KV-cache "slot" from a small pool, so
  multiple requests are handled concurrently instead of queuing behind
  one another.
- `examples/gpt-2/convert-h5-to-ggml.py`: converts a Hugging Face
  `GPT2LMHeadModel` checkpoint to the ggml binary format this server loads.
- `Dockerfile`, `docker-compose.yml`, `Caddyfile`: bakes in the default
  model and runs behind Caddy for automatic Let's Encrypt HTTPS.
- `deploy.md`: step-by-step VPS deployment guide.
- `examples/gpt-2/tests/test_server.py`: black-box test suite against a
  running server (health, both endpoints, streaming, auth, concurrency,
  error handling), stdlib-only.
- `src/`, `include/`, `cmake/`: the underlying [ggml](https://github.com/ggml-org/ggml)
  tensor library this is built on (MIT licensed; see `LICENSE` and `AUTHORS`).

## Default model

Ships with [`MBZUAI/LaMini-GPT-124M`](https://huggingface.co/MBZUAI/LaMini-GPT-124M),
GPT-2 124M fine-tuned on 2.58M distilled instructions, so it actually
follows instructions instead of free-associating like base GPT-2.

**License note:** LaMini-GPT-124M is CC-BY-NC-4.0, non-commercial use
only. Swap in a different checkpoint (see `CHAT_TEMPLATE`/`MODEL_PATH`
below) if you need something commercially usable.

## Quickstart (Docker, recommended)

```bash
git clone <this-repo-url>
cd ggml-gpt2-server

# get the model first, see "Getting the model" below
docker compose up -d --build
```

See [`deploy.md`](deploy.md) for the full VPS walkthrough (DNS, firewall,
`.env` config, sizing, troubleshooting).

## Quickstart (build locally)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gpt-2-server

./build/bin/gpt-2-server -m models/lamini-gpt-124m/ggml-model.bin --port 8080
```

## Getting the model

The converted model isn't committed to git (see `.gitignore`). Convert it
yourself:

```bash
python3 -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MBZUAI/LaMini-GPT-124M',
                   local_dir='models/lamini-gpt-124m',
                   allow_patterns=['*.json', '*.txt', '*.bin'])
"
python examples/gpt-2/convert-h5-to-ggml.py models/lamini-gpt-124m
```

Or run `examples/gpt-2/download-ggml-model.sh 117M` for the original,
non-instruction-tuned base GPT-2 (pairs with `CHAT_TEMPLATE=raw`, see below).

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
| `CTX_SIZE`         | context window (default `1024`, matches the default model)           |
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
The default bundled model (LaMini-GPT-124M) has its own, separate
non-commercial license, see [Default model](#default-model) above.
