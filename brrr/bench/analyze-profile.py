import bisect
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.glob("*rank0*.gz"), key=lambda p: p.stat().st_mtime_ns):
    with gzip.open(path) as f:
        events = json.load(f)["traceEvents"]
    contexts = {}
    for name in ("execute_context_0", "execute_context_1"):
        contexts[name] = sorted((e["ts"], e["ts"]+e["dur"])
            for e in events if e.get("cat") == "gpu_user_annotation"
            and name in e.get("name","") and e.get("dur",0) > 0)
    spans = contexts["execute_context_0"]
    if not spans:
        print(json.dumps(dict(trace=path.name,error="no target execution annotations")))
        continue
    starts = [s for s,_ in spans]
    target = Counter()
    outside = Counter()
    for e in events:
        if e.get("cat") != "kernel":
            continue
        i = bisect.bisect_right(starts,e["ts"])-1
        if i >= 0 and e["ts"] < spans[i][1]:
            target[e["name"]] += e["dur"]
        elif 0 <= i < len(spans)-1:
            outside[e["name"]] += e["dur"]
    assert target and len(spans)>1
    gaps = [b-a for (_,a),(b,_) in zip(spans,spans[1:]) if b>=a]
    print(json.dumps(dict(trace=path.name,graph_calls=len(spans),
        target_median_us=statistics.median(b-a for a,b in spans),
        gap_median_us=statistics.median(gaps) if gaps else None,
        secondary_calls=len(contexts["execute_context_1"]),
        target_top=target.most_common(8), between_target_top=outside.most_common(8))))
