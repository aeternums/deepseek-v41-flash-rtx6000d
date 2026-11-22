import concurrent.futures
import json
from pathlib import Path
import short_chat
import secrets
import time
import urllib.request

base = 'http://127.0.0.1:18136'
model = 'DeepSeek-V4.1-Flash-1017'
marker = 'QZ-' + secrets.token_hex(8)
filler = ' wool lantern harbor snow orchard fence blue meadow.'
prompt = (f'Synthetic mixed test {secrets.token_hex(8)}. Read the following records.\n'
          + filler * 57500 + f'\nThe designated verification code is {marker}.\n'
          + filler * 57500 + '\nFirst return the designated verification code exactly, then write a detailed 10,000-word English implementation checklist for reliable payment webhooks.')
payload = dict(model=model, messages=[dict(role='user', content=prompt)], max_tokens=4096,
               temperature=0, stream=True, stream_options={'include_usage': True},
               chat_template_kwargs={'thinking': False})
req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                             {'Content-Type': 'application/json'})
first = last = None
answer = ''
usage = {}
futures = []


def short(index):
    row = short_chat.run(base, model, 930000 + index)
    return row, time.monotonic()


start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                continue
            event = json.loads(line[5:])
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                content = choice['delta'].get('content') or ''
                if content:
                    last = time.monotonic()
                    if first is None:
                        first = last
                        futures = [pool.submit(short, i) for i in range(8)]
                    answer += content
    long_finished = time.monotonic()
    assert first is not None and len(futures) == 8
    done = [f.result(timeout=180) for f in futures]
assert marker in answer[:160], answer[:200]
assert 1000000 <= usage['prompt_tokens'] <= 1048576 - usage['completion_tokens'], usage
assert max(t for _, t in done) <= long_finished, 'Short batch did not finish inside long decode window'
short_rows = [r for r, _ in done]
short_summary = short_chat.summarize(short_rows, max(t for _, t in done) - first)
result = dict(long_prompt_tokens=usage['prompt_tokens'], long_completion_tokens=usage['completion_tokens'],
              long_ttft=first-start, long_decode_tps=(usage['completion_tokens']-1)/(last-first),
              retrieval_match=True, short_requests=8, fully_overlapped=True, short=short_summary)
Path('mixed-million-result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'short'}), flush=True)
print(json.dumps({k:v for k,v in short_summary.items() if k != 'rows'}), flush=True)
