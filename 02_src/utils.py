import json
import os
from datetime import datetime

def log_to_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data['timestamp'] = datetime.now().isoformat()
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def log_many_to_json(filepath, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        for row in rows:
            payload = dict(row)
            payload['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def read_json_log(filepath, tail=None):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if tail is not None:
        lines = lines[-tail:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
