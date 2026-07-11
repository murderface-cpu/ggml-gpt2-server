# Deploying gpt-2-server to a VPS

This deploys the OpenAI-API-compatible `gpt-2-server` behind Caddy (automatic
HTTPS) using Docker Compose. Two containers: `gpt2-api` (the model server,
not exposed directly) and `caddy` (public entrypoint on 80/443, terminates
TLS, reverse-proxies to `gpt2-api`).

## Prerequisites

- A VPS running Linux (these steps assume Ubuntu; adjust package manager
  commands if different). 2 vCPUs / 2GB RAM is enough for the default
  2-slot configuration. See [Sizing](#sizing) below to scale that up.
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

The model weights (`models/`) are gitignored since they're large binaries and
don't belong in git. Get the repo and the model onto the VPS separately.

**Repo** (commit and push your local changes first if you haven't):

```bash
git clone <your-repo-url> ggml
cd ggml
```

**Model**: copy the already-converted ggml file from your local machine:

```bash
# run this from your LOCAL machine, not the VPS
scp -r "models/lamini-gpt-124m" youruser@your-vps-ip:~/ggml/models/lamini-gpt-124m
```

The Dockerfile expects `models/lamini-gpt-124m/ggml-model.bin` to exist at
build time. The build will fail with a "no such file" error if it's
missing. (Alternative: reconvert on the VPS itself by installing
`requirements.txt` and running `examples/gpt-2/convert-h5-to-ggml.py` there.
That's heavier, since it needs Python/PyTorch/transformers installed, which the
runtime image itself does not need.)

## 3. Configure environment

```bash
cat > .env <<'EOF'
DOMAIN=yourdomain.com
API_KEY=generate-a-long-random-string-here
SLOTS=2
THREADS=2
EOF
```

- **`DOMAIN`**: must match the VPS's DNS A record, or Caddy can't issue a cert.
- **`API_KEY`**: leave blank only if you're fine with the API being open to
  anyone who finds the URL. Recommended to set this for anything public.
  Generate one with `openssl rand -hex 32`.
- **`SLOTS`** / **`THREADS`**: see [Sizing](#sizing).

## 4. Build and start

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

## 5. Verify

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

```bash
git pull
docker compose up -d --build
```

If only the model changed (not the code), re-copy it via `scp` as in step 2,
then `docker compose up -d --build`. Docker only rebuilds the layer that
changed.

## Sizing

Each request in flight needs its own KV-cache slot: ~72MB per slot at the
default `CTX_SIZE=1024`, plus a shared ~315MB for the model weights (fixed,
regardless of `SLOTS`). So total memory is roughly `315MB + SLOTS * 78MB`
(the extra ~6MB per slot is compute-buffer overhead). `THREADS` is the CPU
threads used per in-flight request. `SLOTS * THREADS` shouldn't greatly
exceed the VPS's vCPU count, or concurrent requests will just contend for
the same cores.

| VPS size      | SLOTS | THREADS |
|---------------|-------|---------|
| 1 vCPU / 1GB  | 1     | 1       |
| 2 vCPU / 2GB  | 2     | 2       |
| 4 vCPU / 4GB  | 4     | 2       |

## Troubleshooting

- **Caddy can't get a certificate**: check `docker compose logs caddy`.
  Most common cause: DNS hasn't propagated yet, or `DOMAIN` in `.env`
  doesn't match the A record. Caddy retries automatically once DNS resolves.
- **`docker compose up` fails to bind port 80/443**: something else on the
  VPS is already using it (e.g. a pre-installed Apache/Nginx). Stop it or
  reassign ports.
- **Backend not responding but Caddy is up**: check `docker compose logs gpt2-api`.
  Check the model file actually exists at `models/lamini-gpt-124m/ggml-model.bin`
  before the build ran.
- **Direct backend access for debugging** (bypasses Caddy/TLS): the compose
  file binds `gpt2-api` to `127.0.0.1:8090` on the VPS itself, so
  `ssh youruser@your-vps-ip` then `curl http://localhost:8090/health` works
  without going through DNS/TLS at all. This is not reachable from outside
  the VPS (bound to loopback only).

## License note

The default baked-in model, `MBZUAI/LaMini-GPT-124M`, is **CC-BY-NC-4.0
licensed, non-commercial use only**. If this deployment has any commercial
angle, swap in a permissively-licensed checkpoint instead (see the Dockerfile
header for how to point at a different model).
