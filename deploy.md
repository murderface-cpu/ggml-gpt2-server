# Deploying gpt-2-server to a VPS

This deploys the OpenAI-API-compatible `gpt-2-server` behind Caddy (automatic
HTTPS) using Docker Compose. Two containers: `gpt2-api` (the model server,
not exposed directly) and `caddy` (public entrypoint on 80/443, terminates
TLS, reverse-proxies to `gpt2-api`).

## Prerequisites

- A VPS running Linux (these steps assume Ubuntu; adjust package manager
  commands if different). 2 vCPUs / 2GB RAM is enough for the default
  model and slot count. See [Sizing](#sizing) below for other models.
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
git clone <your-repo-url> ggml
cd ggml
```

## 3. Get a model

The model isn't part of the Docker image; it's mounted from `./models` at
container start, so this is a one-time step you can redo any time to
switch models later (see [Switching models](README.md#switching-models)
in the README).

The easiest path: convert it on your local machine (where you likely
already have the Python venv set up from earlier), then copy the result
over:

```bash
# on your LOCAL machine
python scripts/get_model.py lamini-124m   # or lamini-774m, etc, see README.md#models

scp -r "models/lamini-124m" youruser@your-vps-ip:~/ggml/models/lamini-124m
```

Alternative: run `scripts/get_model.py` directly on the VPS. Works the
same way, but for the `lamini-*` models it needs `pip install -r
requirements.txt` on the VPS first, which pulls in PyTorch. That's a much
heavier install than the runtime image itself needs, so it's usually
better to do the conversion locally and just copy the small output file.

## 4. Configure environment

```bash
cat > .env <<'EOF'
DOMAIN=yourdomain.com
API_KEY=generate-a-long-random-string-here
MODEL_PATH=/models/lamini-124m/ggml-model.bin
CHAT_TEMPLATE=alpaca
SLOTS=2
THREADS=2
EOF
```

- **`DOMAIN`**: must match the VPS's DNS A record, or Caddy can't issue a cert.
- **`API_KEY`**: leave blank only if you're fine with the API being open to
  anyone who finds the URL. Recommended to set this for anything public.
  Generate one with `openssl rand -hex 32`.
- **`MODEL_PATH`** / **`CHAT_TEMPLATE`**: must match whichever model you
  fetched in step 3. `scripts/get_model.py` prints the exact values to use.
- **`SLOTS`** / **`THREADS`**: see [Sizing](#sizing).

## 5. Build and start

```bash
docker compose up -d --build
```

First build takes a few minutes (compiles ggml from source). Watch it:

```bash
docker compose logs -f
```

You should see something like:

```
gpt2-api-1  | main: model loaded, n_ctx = 1024, slots = 2, threads/slot = 2
gpt2-api-1  | main: listening on http://0.0.0.0:8080
caddy-1     | ...certificate obtained successfully...
```

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
python examples/gpt-2/tests/test_server.py \
  --base-url https://yourdomain.com --api-key <your API_KEY>
```

## Updating

**Code changes:**

```bash
git pull
docker compose up -d --build
```

**Switching models** doesn't need a rebuild at all, since the model is
mounted, not baked in:

```bash
python scripts/get_model.py lamini-774m   # on local machine, then scp as in step 3
# update MODEL_PATH / CHAT_TEMPLATE in .env to match
docker compose up -d
```

## Sizing

Each request in flight needs its own KV-cache slot. Numbers below are for
the default `CTX_SIZE=1024`. `THREADS` is the CPU threads used per
in-flight request; `SLOTS * THREADS` shouldn't greatly exceed the VPS's
vCPU count, or concurrent requests will just contend for the same cores.

| Model         | Weights (fixed) | Per slot |
|---------------|------------------|----------|
| `lamini-124m` | ~315MB           | ~72MB    |
| `lamini-774m` | ~1.6GB           | ~360MB   |

So total memory is roughly `weights + SLOTS * per-slot` (plus a few MB per
slot of compute-buffer overhead, small enough to ignore here).

| VPS size      | Model         | SLOTS | THREADS |
|---------------|---------------|-------|---------|
| 1 vCPU / 1GB  | `lamini-124m` | 1     | 1       |
| 2 vCPU / 2GB  | `lamini-124m` | 2     | 2       |
| 4 vCPU / 4GB  | `lamini-124m` | 4     | 2       |
| 2 vCPU / 4GB  | `lamini-774m` | 1     | 2       |
| 4 vCPU / 8GB  | `lamini-774m` | 2     | 2       |

## Troubleshooting

- **Caddy can't get a certificate**: check `docker compose logs caddy`.
  Most common cause: DNS hasn't propagated yet, or `DOMAIN` in `.env`
  doesn't match the A record. Caddy retries automatically once DNS resolves.
- **`docker compose up` fails to bind port 80/443**: something else on the
  VPS is already using it (e.g. a pre-installed Apache/Nginx). Stop it or
  reassign ports.
- **Backend not responding but Caddy is up**: check `docker compose logs gpt2-api`.
  Common cause is `MODEL_PATH` in `.env` pointing at a file that doesn't
  exist under `./models` on the VPS; the server logs a "failed to load
  model" error and exits.
- **Direct backend access for debugging** (bypasses Caddy/TLS): the compose
  file binds `gpt2-api` to `127.0.0.1:8090` on the VPS itself, so
  `ssh youruser@your-vps-ip` then `curl http://localhost:8090/health` works
  without going through DNS/TLS at all. This is not reachable from outside
  the VPS (bound to loopback only).

## License note

The `lamini-*` models are **CC-BY-NC-4.0 licensed, non-commercial use
only**. The `gpt2-*-base` models are the original MIT-licensed OpenAI
weights and don't have that restriction, but aren't instruction-tuned. See
[Models in the README](README.md#models) for the full list and trade-offs.
