#!/usr/bin/env python3
"""
Black-box tests for gpt-2-server (examples/gpt-2/server.cpp) against a
running instance. Talks plain HTTP with the standard library only — no
extra pip installs needed.

Usage:
    # start the server first, e.g.:
    #   ./build/bin/gpt-2-server -m models/gpt-2-117M/ggml-model.bin --port 8080

    python3 examples/gpt-2/tests/test_server.py
    python3 examples/gpt-2/tests/test_server.py --base-url http://vps-ip:8080
    python3 examples/gpt-2/tests/test_server.py --api-key your-secret-key

Exit code is 0 if every test passes, 1 otherwise.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail and not ok:
        line += f" — {detail}"
    print(line)


def request(base_url, method, path, body=None, api_key=None, timeout=60, stream=False):
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def request_json(base_url, method, path, body=None, api_key=None, timeout=60):
    """Returns (status_code, parsed_json_or_None, raw_text)."""
    try:
        with request(base_url, method, path, body=body, api_key=api_key, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        e.close()
        status = e.code
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return status, parsed, raw


def read_sse_events(base_url, path, body, api_key=None, timeout=60):
    """Yields parsed JSON payloads from a text/event-stream response, in order.
    A literal "[DONE]" event yields the string "[DONE]" instead of JSON."""
    with request(base_url, "POST", path, body=body, api_key=api_key, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                yield "[DONE]"
                continue
            yield json.loads(payload)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_health(base_url):
    status, body, raw = request_json(base_url, "GET", "/health")
    ok = status == 200 and body is not None and body.get("status") == "ok"
    report("GET /health", ok, raw)


def test_models(base_url, api_key):
    status, body, raw = request_json(base_url, "GET", "/v1/models", api_key=api_key)
    ok = status == 200 and body is not None and body.get("object") == "list" and len(body.get("data", [])) >= 1
    report("GET /v1/models", ok, raw)


def test_completions(base_url, api_key):
    status, body, raw = request_json(base_url, "POST", "/v1/completions", {
        "prompt": "The quick brown fox",
        "max_tokens": 16,
        "temperature": 0.8,
    }, api_key=api_key)
    ok = (
        status == 200 and body is not None
        and body.get("object") == "text_completion"
        and len(body.get("choices", [])) == 1
        and isinstance(body["choices"][0].get("text"), str)
        and len(body["choices"][0]["text"]) > 0
        and body["choices"][0].get("finish_reason") in ("length", "stop")
        and body.get("usage", {}).get("completion_tokens", 0) > 0
    )
    report("POST /v1/completions (non-streaming)", ok, raw)


def test_completions_streaming(base_url, api_key):
    try:
        chunks = list(read_sse_events(base_url, "/v1/completions", {
            "prompt": "Streaming should work because",
            "max_tokens": 16,
            "stream": True,
        }, api_key=api_key))
    except Exception as e:  # noqa: BLE001
        report("POST /v1/completions (streaming)", False, str(e))
        return

    ok = (
        len(chunks) >= 2
        and chunks[-1] == "[DONE]"
        and all(c == "[DONE]" or c.get("object") == "text_completion.chunk" for c in chunks)
    )
    text = "".join(c["choices"][0]["text"] for c in chunks if c != "[DONE]" and c.get("choices"))
    ok = ok and len(text) > 0
    report("POST /v1/completions (streaming)", ok, f"{len(chunks)} chunks, text={text!r}")


def test_chat_completions(base_url, api_key):
    status, body, raw = request_json(base_url, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 16,
    }, api_key=api_key)
    ok = (
        status == 200 and body is not None
        and body.get("object") == "chat.completion"
        and body.get("choices", [{}])[0].get("message", {}).get("role") == "assistant"
        and isinstance(body["choices"][0]["message"].get("content"), str)
    )
    report("POST /v1/chat/completions (non-streaming)", ok, raw)


def test_chat_completions_streaming(base_url, api_key):
    try:
        chunks = list(read_sse_events(base_url, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 16,
            "stream": True,
        }, api_key=api_key))
    except Exception as e:  # noqa: BLE001
        report("POST /v1/chat/completions (streaming)", False, str(e))
        return

    ok = (
        len(chunks) >= 2
        and chunks[-1] == "[DONE]"
        and chunks[0].get("choices", [{}])[0].get("delta", {}).get("role") == "assistant"
    )
    report("POST /v1/chat/completions (streaming)", ok, f"{len(chunks)} chunks")


def test_missing_prompt(base_url, api_key):
    status, body, raw = request_json(base_url, "POST", "/v1/completions", {"max_tokens": 5}, api_key=api_key)
    ok = status == 400 and body is not None and "error" in body
    report("POST /v1/completions with no prompt -> 400", ok, raw)


def test_invalid_json(base_url, api_key):
    url = base_url.rstrip("/") + "/v1/completions"
    req = urllib.request.Request(url, data=b"{not valid json", method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        status = e.code
        e.close()
    ok = status == 400
    report("POST /v1/completions with malformed JSON -> 400", ok, f"status={status}")


def test_concurrency(base_url, api_key, n=3):
    prompts = [f"Concurrent request #{i}:" for i in range(n)]
    results = [None] * n

    def worker(i):
        status, body, raw = request_json(base_url, "POST", "/v1/completions", {
            "prompt": prompts[i],
            "max_tokens": 20,
        }, api_key=api_key, timeout=90)
        results[i] = (status, body, raw)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    ok = all(
        r is not None and r[0] == 200 and r[1] is not None
        and len(r[1].get("choices", [{}])[0].get("text", "")) > 0
        for r in results
    )
    report(f"{n} concurrent POST /v1/completions", ok, f"elapsed={elapsed:.1f}s")


def test_auth_required(base_url, api_key):
    if not api_key:
        print("[SKIP] auth checks (no --api-key given)")
        return

    # no credentials -> 401
    status, body, raw = request_json(base_url, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 1})
    ok = status == 401
    report("POST /v1/completions without API key -> 401", ok, raw)

    # /health always bypasses auth
    status, body, raw = request_json(base_url, "GET", "/health")
    ok = status == 200
    report("GET /health bypasses API key requirement", ok, raw)

    # correct key -> 200
    status, body, raw = request_json(base_url, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 1}, api_key=api_key)
    ok = status == 200
    report("POST /v1/completions with correct API key -> 200", ok, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081", help="server base URL")
    parser.add_argument("--api-key", default=None, help="API key, if the server was started with one")
    args = parser.parse_args()

    print(f"testing gpt-2-server at {args.base_url}\n")

    test_health(args.base_url)
    test_models(args.base_url, args.api_key)
    test_completions(args.base_url, args.api_key)
    test_completions_streaming(args.base_url, args.api_key)
    test_chat_completions(args.base_url, args.api_key)
    test_chat_completions_streaming(args.base_url, args.api_key)
    test_missing_prompt(args.base_url, args.api_key)
    test_invalid_json(args.base_url, args.api_key)
    test_concurrency(args.base_url, args.api_key)
    test_auth_required(args.base_url, args.api_key)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
