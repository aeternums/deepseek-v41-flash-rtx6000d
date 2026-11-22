import os
from pathlib import Path
import concurrent.futures, json, math, statistics, sys, time, urllib.request

def api(base, path, payload=None):
    headers = {'Content-Type': 'application/json'}
    if keyfile := os.environ.get('GPU8_BENCH_KEY_FILE'):
        headers['Authorization'] = 'Bearer ' + Path(keyfile).read_text().strip()
    req = urllib.request.Request(base+path, None if payload is None else json.dumps(payload).encode(),
                                 headers)
    return urllib.request.urlopen(req, timeout=25)

def run(base, model, i):
    prompt = (f'Synthetic benchmark case {i}. Write a detailed 1,600-word technical guide for '
              'operating a library reservation service. Explain at least twelve numbered scenarios '
              'with examples, including queues, expiry, retries, holidays, and fault recovery.')
    body = dict(model=model, messages=[dict(role='user',content=prompt)], temperature=1, top_p=1,
                max_tokens=512, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking':False})
    # vLLM and Flare use different names for the same no-thinking request.
    if base.endswith((':18020', ':18135', ':18136')):
        body['chat_template_kwargs']={'thinking':False}
    start=time.monotonic()
    first=last=None
    usage=None
    visible=0
    with api(base,'/v1/chat/completions',body) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            if line[5:].strip()==b'[DONE]': break
            event=json.loads(line[5:])
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                d=choice.get('delta',{})
                if d.get('content'):
                    visible+=len(d['content'])
                    last=time.monotonic()
                    if first is None: first=last
                assert not (d.get('reasoning_content') or d.get('reasoning')), 'Unexpected thinking tokens'
    assert first and last>first and visible and usage and usage['completion_tokens']>=128
    return dict(ttft=first-start, decode_tps=(usage['completion_tokens']-1)/(last-first), usage=usage)

def summarize(rows, seconds):
    t=sorted(x['ttft'] for x in rows)
    return dict(requests=len(rows), wall_seconds=seconds, ttft_p50=statistics.median(t),
                ttft_p95=t[math.ceil(.95*len(t))-1],
                decode_tps_p50=statistics.median(x['decode_tps'] for x in rows),
                aggregate_tps=sum(x['usage']['completion_tokens'] for x in rows)/seconds, rows=rows)

if __name__=='__main__':
    if sys.argv[1]=='--self-test':
        x=dict(ttft=1,decode_tps=10,usage=dict(completion_tokens=20))
        assert summarize([x,x],2)['aggregate_tps']==20
        sys.exit()
    base=sys.argv[1]
    with api(base,'/v1/models') as f:
        model=json.load(f)['data'][0]['id']
    for c in map(int,sys.argv[2:]):
        rows=[]
        wall=0
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            for r in range(3):
                t=time.monotonic()
                rows.extend(pool.map(lambda i:run(base,model,10000*c+100*r+i),range(c)))
                wall+=time.monotonic()-t
        print(json.dumps(dict(concurrency=c,**summarize(rows,wall))),flush=True)
