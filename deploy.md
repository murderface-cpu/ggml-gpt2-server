# Deploying llm-server to a VPS

This deploys llama.cpp's own OpenAI-API-compatible server behind Caddy
(automatic HTTPS) using Docker Compose. Two containers: `llm-server` (the
model server, not exposed directly) and `caddy` (public entrypoint on
80/443, terminates TLS, reverse-proxies to `llm-server`).

## Prerequisites

- A VPS running Linux (these steps assume Ubuntu; adjust package manager
  commands if different). The default model (`qwen2.5-1.5b`) needs at
  least 2 vCPUs / 2GB RAM at the default context size and slot count.
  See [Sizing](#sizing) below.
- A domain name with an **A record pointing at the VPS's public IP**.
  Caddy needs this to obtain a Let's Encrypt certificate. It cannot get
  one for a bare IP address.
- Ports 80 and 443 open to the internet on the VPS.
- SSH access to the VPS.

## 1. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
```

Verify:

```bash
docker compose version
```

## 2. Get the code onto the VPS

```bash
git clone <your-repo-url> llm-server
cd llm-server
```

## 3. Get a model

The model isn't part of a Docker image; it's mounted from `./models` at
container start, so this is a one-time step you can redo any time to
switch models later (see [Switching models](README.md#switching-models)
in the README).

```bash
# on your LOCAL machine (or directly on the VPS, doesn't matter here —
# unlike the old HF-checkpoint-conversion pipeline, this is just a single
# file download, no PyTorch/heavy ML deps needed)
pip install huggingface_hub
python scripts/get_model.py qwen2.5-1.5b   # or qwen2.5-0.5b for a smaller VPS

# if you downloaded it locally, copy it over:
scp -r "models/qwen2.5-1.5b" youruser@your-vps-ip:~/llm-server/models/qwen2.5-1.5b
```

## 4. Configure environment

```bash
cat > .env <<'EOF'
DOMAIN=yourdomain.com
API_KEY=generate-a-long-random-string-here
MODEL_PATH=/models/qwen2.5-1.5b/model.gguf
CTX_SIZE=8192
SLOTS=1
THREADS=2
EOF
```

- **`DOMAIN`**: must match the VPS's DNS A record, or Caddy can't issue a cert.
- **`API_KEY`**: leave blank only if you're fine with the API being open to
  anyone who finds the URL. Recommended to set this for anything public.
  Generate one with `openssl rand -hex 32`.
- **`MODEL_PATH`**: must match whichever model you fetched in step 3.
- **`CTX_SIZE`** / **`SLOTS`** / **`THREADS`**: see [Sizing](#sizing).

## 5. Start

```bash
docker compose up -d
```

No build step: this pulls the official `ghcr.io/ggml-org/llama.cpp:server`
image. Watch startup:

```bash
docker compose logs -f
```

You should see the model load and the server start listening, followed by
Caddy obtaining a certificate.

## 6. Verify

```bash
curl https://yourdomain.com/health
# {"status":"ok"}

curl https://yourdomain.com/v1/chat/completions \
  -H "Authorization: Bearer <your API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'
```

Or run the test suite from your local machine against the live deployment:

```bash
python tests/test_server.py \
  --base-url https://yourdomain.com --api-key <your API_KEY>
```

## Updating

**Compose/Caddy config changes:**

```bash
git pull
docker compose up -d
```

**New llama.cpp version**: `docker compose pull && docker compose up -d`
picks up the latest `server` image tag.

**Switching models** doesn't need a rebuild or pull, since the model is
mounted, not baked in:

```bash
python scripts/get_model.py qwen2.5-0.5b   # then scp as in step 3 if run locally
# update MODEL_PATH in .env to match
docker compose up -d
```

## Sizing

Total memory is roughly `weights + SLOTS * per-slot` (KV cache scales with
`CTX_SIZE`; numbers below are for llama.cpp's default f16 KV cache).
`THREADS` is the CPU threads used per in-flight request; `SLOTS * THREADS`
shouldn't greatly exceed the VPS's vCPU count, or concurrent requests will
just contend for the same cores.

| Model          | Weights (fixed) | Per slot @ CTX_SIZE=8192 | Per slot @ 32768 |
|----------------|------------------|----------------------------|---------------------|
| `qwen2.5-0.5b` | ~500MB           | ~96MB                     | ~384MB               |
| `qwen2.5-1.5b` | ~1.1GB           | ~224MB                    | ~896MB               |

| VPS size      | Model          | CTX_SIZE | SLOTS | THREADS |
|---------------|----------------|----------|-------|---------|
| 1 vCPU / 1GB  | `qwen2.5-0.5b` | 4096     | 1     | 1       |
| 2 vCPU / 2GB  | `qwen2.5-1.5b` | 8192     | 1     | 2       |
| 4 vCPU / 4GB  | `qwen2.5-1.5b` | 8192     | 2     | 2       |
| 4 vCPU / 8GB  | `qwen2.5-1.5b` | 32768    | 2     | 2       |

## Troubleshooting

- **Caddy can't get a certificate**: check `docker compose logs caddy`.
  Most common cause: DNS hasn't propagated yet, or `DOMAIN` in `.env`
  doesn't match the A record. Caddy retries automatically once DNS resolves.
- **`docker compose up` fails to bind port 80/443**: something else on the
  VPS is already using it (e.g. a pre-installed Apache/Nginx). Stop it or
  reassign ports.
- **Backend not responding but Caddy is up**: check `docker compose logs
  llm-server`. Common cause is `MODEL_PATH` in `.env` pointing at a file
  that doesn't exist under `./models` on the VPS.
- **Direct backend access for debugging** (bypasses Caddy/TLS): the compose
  file binds `llm-server` to `127.0.0.1:8090` on the VPS itself, so
  `ssh youruser@your-vps-ip` then `curl http://localhost:8090/health` works
  without going through DNS/TLS at all. This is not reachable from outside
  the VPS (bound to loopback only).

## License note

The `qwen2.5-*` models are Apache-2.0, usable commercially. See
[Models in the README](README.md#models) for details.
