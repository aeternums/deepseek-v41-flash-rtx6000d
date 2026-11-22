"""Compare warm long-context chat at C1/C8 on an isolated GPU8 endpoint."""
import concurrent.futures
import json
import sys
import time
from urllib.request import Request, urlopen

MODEL = "DeepSeek-V4.1-Flash-1017"
MARKER = "QZ-48271"
ROW = " wool lantern harbor snow orchard fence blue meadow."
OPTIONS = {"thinking": False, "enable_thinking": False}


def messages(size, question):
    return [
        {"role": "user", "content": "Archive records:" + ROW * (size // 9 // 2)
         + f" The archive reference is {MARKER}." + ROW * (size // 9 // 2)},
        {"role": "assistant", "content": "I have read the archive."},
        {"role": "user", "content": question},
    ]


def post(base, path, body):
    return urlopen(Request(base + path, json.dumps(body).encode(),
                           {"Content-Type": "application/json"}), timeout=300)


def run(base, history, limit):
    body = dict(model=MODEL, messages=history, chat_template_kwargs=OPTIONS,
                temperature=0, seed=20261121, max_tokens=limit, stream=True,
                stream_options={"include_usage": True})
    start = time.monotonic()
    first = last = None
    usage = None
    answer = []
    ended = False
    with post(base, "/v1/chat/completions", body) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                ended = True
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event["choices"]:
                content = choice["delta"].get("content")
                if content:
                    last = time.monotonic()
                    first = first or last
                    answer.append(content)
    assert ended and first and usage, "Incomplete SSE response"
    count = usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft=first-start, seconds=time.monotonic()-start,
                decode_tps=(count-1)/(last-first) if last > first else None,
                input=usage["prompt_tokens"], cached=cached, output=count,
                marker=MARKER in "".join(answer)[:160])


def main():
    mode, base = sys.argv[1:]
    if mode == "tokenize":
        for size in (100000, 200000):
            with post(base, "/tokenize", dict(model=MODEL, messages=messages(size, "Read the archive."),
                                             chat_template_kwargs=OPTIONS)) as response:
                count = json.load(response)["count"]
            assert abs(count-size) < size*.02, (size,count)
            print(json.dumps(dict(target=size, actual=count)))
        return
    assert mode == "bench" and base == "http://127.0.0.1:18136", "Use the isolated research endpoint"
    for size in (100000, 200000):
        warm = run(base, messages(size, "Return only the archive reference."), 32)
        print(json.dumps(dict(size=size, warm=warm)), flush=True)
        assert warm["marker"], warm
        for concurrency in (1, 8):
            for repeat in range(3):
                start = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    rows = list(pool.map(
                        lambda i: run(base, messages(size,
                            f"Follow-up {concurrency}-{repeat}-{i}. Start with the exact archive reference. "
                            "Then write a detailed practical guide to organizing and verifying archived records. "
                            "Include at least 12 examples and explain the tradeoffs in plain language."), 512),
                        range(concurrency)))
                elapsed = time.monotonic()-start
                print(json.dumps(dict(size=size, concurrency=concurrency, repeat=repeat,
                    elapsed=elapsed, aggregate_tps=sum(x["output"] for x in rows)/elapsed, rows=rows)), flush=True)
                assert all(x["marker"] and x["cached"] is not None and x["cached"] >= .95*x["input"]
                           and x["output"] >= 128 for x in rows), rows
        # Independent 200K x 8 exceeds the production KV pool. Use C4 at 200K.
        concurrency = 8 if size == 100000 else 4
        def independent(i, question):
            history = messages(size, question)
            history[0]["content"] = f"Independent archive {i}: " + history[0]["content"]
            return history

        for i in range(concurrency):
            warm = run(base, independent(i, "Return only the archive reference."), 32)
            assert warm["marker"], warm
        for repeat in range(3):
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                rows = list(pool.map(lambda i: run(base, independent(i,
                    f"Follow-up {repeat}. Start with the exact archive reference. "
                    "Then write a detailed practical guide to organizing and verifying archived records. "
                    "Include at least 12 examples and explain the tradeoffs in plain language."), 512),
                    range(concurrency)))
            elapsed = time.monotonic()-start
            print(json.dumps(dict(size=size, prefix="independent", concurrency=concurrency,
                repeat=repeat, elapsed=elapsed, aggregate_tps=sum(x["output"] for x in rows)/elapsed,
                rows=rows)), flush=True)
            assert all(x["marker"] and x["cached"] is not None and x["cached"] >= .95*x["input"]
                       and x["output"] >= 128 for x in rows), rows


if __name__ == "__main__":
    main()
