"""Strict, bounded reader of metadata and selected cpu_op events.
It never decodes stack events or whole 16GB Chrome traces. Only minimal
step-marker `name`, `ts`, `dur`, `tid`, and `args` keys are retained.
"""
from pathlib import Path
import json, hashlib, argparse, bisect, collections
WANTED = [b'"name": "execute_new_', b'"name": "Iteration Start: PyTorch Profiler"', b'"name": "Record Window End"']
START = b'\n  {'

def extract(path):
    chosen=[]; counts=collections.Counter(); bad=0; chunk_size=128*1024*1024
    hashes=[]
    with path.open('rb') as f:
        total=0; offset=0
        while data:=f.read(chunk_size):
            total+=len(data)
            if total<=16777216: hashes.append(hashlib.sha256(data).digest())
            if total%(1024*1024*1024)<chunk_size:print('boundary scan',path.name,total,'selected',len(chosen),flush=True)
            candidates=[at for marker in WANTED for at in find_all(marker,data)]
            for at in sorted(candidates):
                start=data.rfind(START,0,at)
                end=data.find(START,at)
                if start<0 or end<0:continue
                body=data[start+1:end].strip()
                if body.endswith(b','):body=body[:-1]
                try:obj=json.loads(body)
                except ValueError:bad+=1;continue
                if 'args' in obj and isinstance(obj['args'],dict):obj['args']={k:v for k,v in obj['args'].items()if k in ('External id','Steps','stream','device')}
                chosen.append(obj)
    # Must not produce many cpu_op / user_annotation else abort rather than
    # accidentally returning an enormous host trace as report.
    assert len(chosen)<1000,(len(chosen),collections.Counter(d['name']for d in chosen))
    return {'file':path.name,'size':path.stat().st_size,'events':chosen,'skipped':bad,'prefix_chunk_hashes':hashes}

def find_all(marker,data):
    p=0
    try:
        while 1:
            p=data.index(marker,p);yield p;p+=len(marker)
    except ValueError:return

def main():
    p=argparse.ArgumentParser()
    p.add_argument('file',type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    record=extract(a.file)
    events=sorted(record['events'],key=lambda e:e['ts'])
    grouped=collections.defaultdict(list)
    for event in events:grouped[event['name']].append(event)
    summary={name:{'count':len(rows),'first_ts':min(e['ts']for e in rows),'last_ts':max(e['ts']for e in rows),'duration_sum_ms':sum(e.get('dur',0)for e in rows)/1000,'duration_ms':[e.get('dur',0)/1000 for e in rows]}for name,rows in grouped.items()}
    report={'file':record['file'],'raw_scanned_bytes':record['size'],'skipped':record['skipped'],'events':events,'summary':summary,'warning':'Cuda graph annotation duration overcounts GPU duration through sibling overlap; only timestamps describe scheduling here.'}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(a.out),'bytes':a.out.stat().st_size,'event_count':len(events),'summary':summary},indent=2))

if __name__=='__main__':main()
