"""Very efficient partial Chrome-traces reader for GPU-kernel subsection only.

Do not load multi-GB full stack/CPU event trace as whole JSON. Byte-search
`"cat": "kernel"` first; then parse only its few adjacent object boundaries.
Each selected `name`, `ts`, `dur`, `args` (few bytes) retained in small list.
The destined result must be tiny, and originate from *complete* trace files.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import pathlib


def union_ms(events):
    spans = sorted((e['ts'], e['ts']+e.get('dur', 0))for e in events if e.get('ts')is not None and e.get('dur',0)>0)
    if not spans:return 0
    total=0;first,last=spans[0]
    for start,end in spans[1:]:
        if start>last:total+=last-first;first,last=start,end
        else:last=max(last,end)
    return (total+last-first)/1000


def extract(path):
    if not path.name.endswith('.json') or path.name.endswith('.tmp'):
        raise ValueError(f'Not completed trace: {path}')
    found=[]
    marker=b'"cat": "kernel"'
    event_start=b'\n  {'
    carry=b''
    chunk_size=64*1024*1024
    total=bad=0
    target_count=0
    with path.open('rb',buffering=chunk_size) as f:
        while raw:=f.read(chunk_size):
            total+=len(raw)
            data=carry+raw
            start=0;after=0;carry_from=0
            while (at:=data.find(marker,after))>=0:
                after=at+len(marker)
                before=data.rfind(event_start,0,at)
                end=data.find(event_start,at)
                if before<0 or end<0:
                    # Retain prefix through partial end marker: a root event may
                    # straddle the next byte-read chunk.
                    if before>=0:carry_from=before
                    break
                carry_from=end
                chunk=data[before+1:end].strip()
                # Root array entries may end with a comma. Only these short
                # kernel objects are accepted no Python-stack nesting.
                if chunk.endswith(b','):chunk=chunk[:-1]
                try:
                    row=json.loads(chunk)
                except json.JSONDecodeError:
                    bad+=1;continue
                assert row.get('cat')=='kernel'
                assert isinstance(row.get('name'),str)
                assert isinstance(row.get('ts'),(float,int))
                assert isinstance(row.get('dur'),(float,int))
                found.append(row);target_count+=1
            carry=data[carry_from:] if carry_from else data[-1048576:]
            total_reported=json.dumps({'file':path.name,'bytes_scanned':total,'kernels':len(found),'skipped':bad})
            if total%(512*1024*1024)<chunk_size:print('scan',total_reported,flush=True)
    return {'file':path.name,'size':path.stat().st_size,'raw_scanned_bytes':total,
        'kernel_events':found,'excluded_bad_records':bad}


def classify(name):
    l=name.lower()
    if 'cross_device_reduce' in l or 'ncclDevKernel'.lower() in l:return 'communication/TP allreduce or allgather'
    if 'fused_moe_kernel' in l:return 'expert fused GEMM'
    if 'moe' in l or 'per_token_group_quant' in l or 'index_elementwise' in l:return 'expert route/quant/align/support'
    if 'flash' in l or 'rope' in l:return 'attention/rope'
    if 'nvjet' in l or 'gemm' in l or 'splitkreduce' in l:return 'other linear GEMM'
    return 'other GPU kernel'


def analyze(record):
    kernels=record['kernel_events']
    begin=min(x['ts']for x in kernels);end=max(x['ts']+x['dur']for x in kernels)
    rows={}
    for group in sorted(set(classify(x['name'])for x in kernels)):
        matching=[x for x in kernels if classify(x['name'])==group]
        rows[group]={'count':len(matching),'raw_sum_ms':sum(x['dur']for x in matching)/1000,'union_ms':union_ms(matching)}
    names=collections.defaultdict(list)
    for e in kernels:names[e['name']].append(e)
    return {k:v for k,v in record.items()if k!='kernel_events'}|{'union_all_streams_ms':union_ms(kernels),'gpu_span_ms':(end-begin)/1000,'categories':rows,'named_kernels':[{'name':name,'count':len(items),'sum_ms':sum(x['dur']for x in items)/1000} for name,items in sorted(names.items(),key=lambda x:sum(y['dur']for y in x[1]),reverse=True)[:40]]}


def main():
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);a=p.parse_args()
    parsed=[extract(path)for path in a.files]
    os.makedirs(a.out.parent,exist_ok=True)
    a.out.write_text(json.dumps({'rank_analysis':[analyze(x)for x in parsed],'overlap_caveat':'active union is device-wide; step-only boundaries need execute_new timestamps','profiler_reference':'full stack / shapes generates extreme CPU overhead and must not measure service speed','raw_trace_files_retained_in_pod':'/tmp/m2-decode-profile-1'},indent=2)+'\n')
    print(json.dumps({'output':str(a.out),'selected':[{'file':x['file'],'kernels':len(x['kernel_events'])}for x in parsed]},indent=2),flush=True)
    print(json.dumps([analyze(x)for x in parsed],indent=2),flush=True)


if __name__=='__main__':main()
