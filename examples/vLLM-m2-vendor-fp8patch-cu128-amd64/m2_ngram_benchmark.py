"""Read-only, bounded M2 loopback A/B against an explicit existing EP8 pod.

No proxy/gateway full-response payload is persisted. Warmup then measured
fixed-256-token requests on TP8+EP8 at concurrency 1, 2, 8. The repetitive
prompt gives n-gram a fair repetition workload. Workload-local counters exclude
warmups and starts after the earlier measured request groups.
Boundary only; no full stream request/reason/block text goes to persistent file.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import time
import urllib.request

PROMPT = 'Write three concise bullet points explaining how expert parallelism improves mixture-of-experts inference.'
REPEAT_PROMPT = ('Repeat the following four-line block verbatim exactly ten times, with no other text: '
    'alpha beta gamma delta. alpha beta gamma delta alpha. '
    'beta gamma gamma delta numeric 497. '
    'quantized experts route through shared communication.')


def request(url, payload=None, timeout=180, **ignored):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body,
        headers={'Content-Type': 'application/json'} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return {'status': response.status, 'data': json.load(response)}


def run_acceptance(url):
    simple = {'model': 'M2', 'max_tokens': 600, 'temperature': 0,
        'messages': [{'role': 'user', 'content': 'Reply only: M2 works'}],
        'chat_template_kwargs': {'reasoning_effort': 'low'}}
    with urllib.request.urlopen(urllib.request.Request(url + '/chat/completions',
        data=json.dumps(simple).encode(), headers={'Content-Type': 'application/json'}),
        timeout=240) as response:
        result = json.load(response)
    choice = result['choices'][0]
    assert choice['finish_reason'] == 'stop', choice
    assert 'M2 works' in (choice['message'].get('content') or '')
    tool_payload = {'model': 'M2', 'max_tokens': 700, 'temperature': 0,
        'messages': [{'role': 'user', 'content': 'Call get_weather for Abu Dhabi directly. Output actual function invocation, not surrounding discussion.'}],
        'tool_choice': 'auto',
        'tools': [{'type': 'function', 'function': {'name': 'get_weather',
            'description': 'Current weather for a city', 'parameters': {'type': 'object',
            'properties': {'location': {'type': 'string'}}, 'required': ['location']}}}],
        'chat_template_kwargs': {'reasoning_effort': 'low'}}
    with urllib.request.urlopen(urllib.request.Request(url + '/chat/completions',
        data=json.dumps(tool_payload).encode(), headers={'Content-Type': 'application/json'}),
        timeout=240) as response:
        tool = json.load(response)
    tool_choice = tool['choices'][0]
    assert tool_choice['finish_reason'] == 'tool_calls'
    function = next(x['function'] for x in tool_choice['message']['tool_calls'] if x['type'] == 'function')
    assert function['name'] == 'get_weather'
    assert json.loads(function['arguments'])['location'] == 'Abu Dhabi'
    return {'chat_finish': choice['finish_reason'], 'tool': tool,
            'token_details': {'chat_prompt_tokens': result['usage']['prompt_tokens'],
                              'chat_completion_tokens': result['usage']['completion_tokens']}}


def metric(counters=None):
    names = ['vllm:generation_tokens_total']
    if counters:
        names.extend([
            'vllm:spec_decode_num_drafts_total',
            'vllm:spec_decode_num_draft_tokens_total',
            'vllm:spec_decode_num_accepted_tokens_total',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total'])
    text = urllib.request.urlopen('http://127.0.0.1:18000/metrics', timeout=15).read().decode()
    values = {}
    position = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        key, _, raw_value = line.rpartition(' ')
        name = key.partition('{')[0]
        if name in ('vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_draft_tokens_total',
                   'vllm:spec_decode_num_accepted_tokens_total', 'vllm:generation_tokens_total'):
            values[name] = values.get(name, 0) + float(raw_value)
        elif name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            label = next((part for part in key.split(',') if part.startswith('position=')), 'position=unknown')
            position[label] = position.get(label, 0) + float(raw_value)
    values['accepted_by_position'] = position
    return values


def difference_after(before, after):
    result = {}
    for key, val in after.items():
        if isinstance(val, dict):
            result[key] = {p: x - before[key].get(p, 0) for p, x in val.items()}
        else:
            result[key] = val - before.get(key, 0)
    drafts = result.get('vllm:spec_decode_num_drafts_total', 0)
    drafted = result.get('vllm:spec_decode_num_draft_tokens_total', 0)
    accepted = result.get('vllm:spec_decode_num_accepted_tokens_total', 0)
    result['mean_acceptance_length_including_bonus'] = 1 + accepted / drafts if drafts else None
    result['token_acceptance_ratio'] = accepted / drafted if drafted else None
    return result


def timed_one(a, url, prompt, estimated_tokens=256):
    payload = {'model': 'M2', 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': estimated_tokens, 'temperature': 0}
    start = time.monotonic()
    data = a.request(url + '/chat/completions', payload, timeout=180)
    elapsed = time.monotonic() - start
    response = data['data']
    usage = response['usage']
    choice = response['choices'][0]
    return {'elapsed_s': elapsed, 'completion_tokens': usage['completion_tokens'],
        'prompt_tokens': usage['prompt_tokens'], 'finish': choice['finish_reason'],
        'generated_tokens_per_s': usage['completion_tokens'] / elapsed,
        'content_reply_eos_checked_success': choice['finish_reason'] in ('length', 'stop'),
        'output_sha256': hashlib.sha256(((choice['message'].get('content') or '').encode()
            + (choice['message'].get('reasoning') or '').encode())).hexdigest()}


def measure_workload(a, name, prompt, repeat, concurrency):
    # Both controlled prompts generate exactly 256 tokens; compare within-
    # workload deltas only. Metrics include no initialization or acceptance.
    counter_values = name in ('fixed_reference', 'repetitive_generation')
    for _ in range(2):
        timed_one(a, 'http://127.0.0.1:18000/v1', prompt)
    singles = []
    for _ in range(repeat):
        before = metric(counter_values)
        row = timed_one(a, 'http://127.0.0.1:18000/v1', prompt)
        after = metric(counter_values)
        row['counters'] = difference_after(before, after)
        singles.append(row)
    assert all(r['finish'] == 'length' and r['completion_tokens'] == 256 for r in singles)
    unique_prompts = {r['prompt_tokens'] for r in singles}
    assert len(unique_prompts) == 1
    records = {}
    for c in concurrency:
        before = metric(counter_values)
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            rows = list(executor.map(lambda _: timed_one(a, 'http://127.0.0.1:18000/v1', prompt), range(c)))
        elapsed = time.monotonic() - start
        after = metric(counter_values)
        tokens = [row['completion_tokens'] for row in rows]
        metrics_delta = difference_after(before, after)
        for key in ('vllm:generation_tokens_total', 'vllm:spec_decode_num_drafts_total',
                    'vllm:spec_decode_num_draft_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total'):
            if key in metrics_delta:
                assert metrics_delta[key] >= 0, (key, metrics_delta[key])
        if counter_values:
            # Per-request exact output accounting is the throughput denominator;
            # the asynchronously logged generation counter can include other
            # admin/startup probes (the supplied custom startup wrapper probes
            # API upstreams) and vLLM may write it beyond a request finish event.
            # Record mismatch explicitly instead of erasing mixed-source data.
            assert metrics_delta['vllm:generation_tokens_total'] >= c * 256, (
                'counter cannot be less than request outputs', metrics_delta)
            assert all(row['content_reply_eos_checked_success'] for row in rows)
        records[str(c)] = {'wall_s': elapsed, 'generated_tokens_per_s': sum(tokens) / elapsed,
            'total_tokens': sum(tokens), 'per_request': rows, 'counters': metrics_delta,
            'expected_gen_from_completed_requests': c * 256,
            'counter_skew': metrics_delta.get('vllm:generation_tokens_total', c * 256) - c * 256,
            'metric_confinement_note': 'Private /metrics can lag/include healthy endpoint warmups; actual timed completed-request token sums are rate basis even when local counter delta differs.',
            'speculative_stats_after': after}
    return {'prompt': prompt, 'prompt_tokens': unique_prompts.pop(),
        'max_completion_tokens': 256, 'temperature': 0, 'warmups': 2,
        'singles': singles,
        'single_median_tps': statistics.median(r['generated_tokens_per_s'] for r in singles),
        'single_median_request_latency_s': statistics.median(r['elapsed_s'] for r in singles),
        'parallel': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod-uid', required=True)
    parser.add_argument('--ngram-k', required=True, type=int, choices=[0, 1, 2])
    parser.add_argument('--output', required=True)
    parser.add_argument('--with-repeat', action='store_true',
        help='run second optimized workload with prompt likely triggering actual accepted ngram')
    parser.add_argument('--full-two-eight', action='store_true')
    args = parser.parse_args()
    # Bind output to the observed fresh pod, never to its short-lived previous UID.
    pod_uid_path = pathlib.Path('/tmp/m2-ngram-pod-uid')
    assert pod_uid_path.is_file(), 'stage verified pod UID into /tmp after fresh readiness'
    assert pod_uid_path.read_text().strip() == args.pod_uid
    u = 'http://127.0.0.1:18000/v1'
    with urllib.request.urlopen(u + '/models', timeout=20) as response:
        models = json.load(response)['data']
    assert len(models) == 1 and models[0]['id'] == 'M2' and models[0]['max_model_len'] == 262144
    acceptance = run_acceptance(u)
    initial = metric(args.ngram_k > 0)
    result = {'pod_uid': args.pod_uid, 'ngram_k': args.ngram_k,
        'protocol': 'TP8-ranked original vendor with no profile; fixed loopback temp0 same universe A/B',
        'initial_counters': initial,
        'model_eos_and_tool_acceptance': {'eos_native_stop': True,
            'm2_response_correct': True, 'tool_location_correct': True,
            'tool_finish_reason': acceptance['tool']['choices'][0]['finish_reason'],
            'chat_usage': acceptance['token_details']}}
    start = time.monotonic()
    import types
    a = types.SimpleNamespace(request=request)
    result['fixed_reference'] = measure_workload(a, 'fixed_reference', PROMPT, 3, [1, 2, 8] if args.full_two_eight else [1, 8])
    if args.with_repeat:
        result['repetitive_generation'] = measure_workload(a, 'repetitive_generation', REPEAT_PROMPT, 3, [1, 2] if args.full_two_eight else [1])
    result.update(total_benchmark_wall_s=time.monotonic() - start, all_checks_passed=True,
                  final_counters=metric(args.ngram_k > 0))
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    summary = {'ngram_k': args.ngram_k, 'checks': result['model_eos_and_tool_acceptance'],
        'total_benchmark_wall_s': result['total_benchmark_wall_s']}
    for label, workload in [('reference', result['fixed_reference']),
                             ('repeat', result.get('repetitive_generation'))]:
        if workload:
            summary[label] = {'single_median_tps': workload['single_median_tps'],
                'single_draft_cumulative': sum(r['counters'].get('vllm:spec_decode_num_accepted_tokens_total', 0) for r in workload['singles']),
                'parallel': {c: {'throughput': v['generated_tokens_per_s'], 'wall_s': v['wall_s'],
                    'drafted': v['counters'].get('vllm:spec_decode_num_draft_tokens_total'),
                    'accepted': v['counters'].get('vllm:spec_decode_num_accepted_tokens_total'),
                    'mean_chain': v['counters'].get('mean_acceptance_length_including_bonus')}
                    for c, v in workload['parallel'].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
