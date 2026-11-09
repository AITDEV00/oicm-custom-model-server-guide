"""Summarize bounded per-rank PyTorch Chrome traces from actual M2 decode.

Keep full trace files separate. Select one rank device duration, duplicating
eight-rank parallel time is invalid. Attribute best-known CUDA stream busy
union, not GPU time indiscriminately or infer dispatch from a collective name.
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
import pathlib
import re
import statistics


def union_duration(events):
    """Return how much union interval time is busy, preserving overlap union."""
    times = sorted((e['ts'], e['ts'] + e['dur']) for e in events if e.get('dur', 0) > 0)
    total = 0.0
    if not times:
        return total
    low, high = times[0]
    for start, end in times[1:]:
        if start > high:
            total += high - low
            low, high = start, end
        else:
            high = max(high, end)
    return total + high - low


def category(name):
    low = name.lower()
    if low.startswith('nccl') or 'nccl_' in low:
        return 'NCCL collective (not automatically EP dispatch)'
    if 'fused_moe_kernel' in low or 'fusedmoe' in low:
        return 'FP8 fused expert GEMM'
    if 'moe_align' in low or 'moe_permute' in low or 'moe_unpermute' in low:
        return 'MoE align/permute/routing support'
    if 'dynamic_scaled_int8_quant' in low or 'per_token' in low and 'quant' in low or 'group_quant' in low:
        return 'MoE FP8 dynamic activation quantization'
    if 'topk' in low or 'top_k' in low or 'moe_sum' in low:
        return 'MoE router/top-k/reduction'
    if 'flash' in low and ('attn' in low or 'fwd' in low):
        return 'Attention FlashAttention'
    if 'rope' in low or 'pos_embed' in low:
        return 'Attention RoPE'
    if 'rmsnorm' in low or '_norm' in low:
        return 'Norm'
    if 'gemm' in low or 'cutlass' in low or 'triton_persistent_tma' in low or 'sm90' in low or 'sm100' in low:
        return 'Other linear GEMM/kernel'
    if 'memcpy' in low.lower() or 'memset' in low.lower():
        return 'Memory copy/fill'
    return 'Other GPU kernel'


def analyze(path):
    trace = json.loads(path.read_text())
    events = trace.get('traceEvents', trace if isinstance(trace, list) else [])
    kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memset', 'gpu_memcpy')]
    cpu_steps = [e for e in events if e.get('cat') in ('cpu_op','user_annotation')
                 and e.get('name', '').startswith('execute_new_')]
    categories = collections.defaultdict(lambda: {'count': 0, 'duration_us': 0.0, 'exclusive_union_us': 0.0})
    for e in kernels:
        key = category(e.get('name', ''))
        categories[key]['count'] += 1
        categories[key]['duration_us'] += e.get('dur', 0)
    kernel_union = union_duration(kernels)
    annotations = [{'name':e['name'],'ts':e['ts'],'dur':e.get('dur',0)} for e in events if e.get('cat')=='user_annotation']
    for key in categories:
        categories[key]['exclusive_union_us'] = union_duration([e for e in kernels if category(e.get('name', '')) == key])
    times = [e['ts'] for e in kernels if e.get('ts') is not None]
    record = {'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'all_trace_events':len(events),'gpu_kernel_count':len(kernels),
              'gpu_time_union_ms':kernel_union/1000,
              'gpu_wall_span_ms':((max((e['ts']+e.get('dur',0)for e in kernels),default=0)-min(times,default=0))/1000),
              'categories':dict(sorted(categories.items(),key=lambda kv:kv[1]['duration_us'],reverse=True)),
              'cpu_model_steps':cpu_steps,
              'annotations':[{'cat':e.get('cat'), 'name':e['name'],'ts':e['ts'],'dur':e.get('dur',0), 'args':e.get('args')}for e in events if e.get('cat') in ('cpu_instant_event', 'user_annotation', 'gpu_user_annotation')],
              'cuda_graph_launches':[{'name':e['name'],'ts':e['ts'],'dur':e.get('dur',0),'args':e.get('args')} for e in events if 'graphlaunch' in e.get('name','').lower()],
              'gpu_kernels_by_stream':dict(collections.Counter(str(e.get('args',{}).get('stream',e.get('args',{}).get('stream id','none'))) for e in kernels)),
              'top_kernels':[]}
    tops = collections.defaultdict(lambda: {'count':0,'duration_ms':0.0})
    for e in kernels:
        tops[e['name']]['count']+=1; tops[e['name']]['duration_ms']+=e.get('dur',0)/1000
    record['top_kernels']=[{'name':name,**v}for name,v in sorted(tops.items(),key=lambda kv:kv[1]['duration_ms'],reverse=True)[:80]]
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=pathlib.Path)
    parser.add_argument('--out',required=True,type=pathlib.Path)
    args=parser.parse_args()
    paths = sorted(args.directory.glob('*.pt.trace.json'))
    assert paths, 'Expected rank-trace-only directory or filter a report elsewhere'
    result={'traces':[analyze(p)for p in paths],
            'limitations':['Sum durations is by kernel, with concurrent-stream overlap included. Compare with union for actual rank device busy.',
                           'NCCL names do not prove MoE EP all-to-all here (DP1 fused method); TP model-attention is also reduced.',
                           'Profiler overhead and exact request edges must be annotated from the bounded API response.',
                           'Do not add rank0 plus ranks1-7 to infer wall propagation delay. Rank0 direct display same-wave kernels often occur concurrently']}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    for item in result['traces']:
        print(item['file'],'kernels',item['gpu_kernel_count'],'union_ms',item['gpu_time_union_ms'])
        print('TOP_KERNEL_CATEGORIES',json.dumps(item['categories'],indent=2))
        print('TOP_KERNELS',json.dumps(item['top_kernels'][:20],indent=2))


if __name__=='__main__':
    main()
