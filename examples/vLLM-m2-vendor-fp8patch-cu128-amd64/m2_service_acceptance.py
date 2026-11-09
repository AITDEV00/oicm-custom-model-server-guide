"""Read-only M2 service acceptance and small repeat-baseline measurements.

Runs only after the independently validated original worker becomes ready.
Requires --url, --transcript via filename (no secrets), and a secret supplied
via M2_GATEWAY_TOKEN if a private HTTPS gateway bears authorization.
No model keys/proven persistent logs include this header. Never writes PVC.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, pathlib, ssl, time
import http.client, urllib.parse


def valid_url(value):
    parsed = urllib.parse.urlsplit(value)
    assert parsed.scheme in ('http', 'https') and parsed.netloc
    return value.rstrip('/')


def request(url, payload=None, timeout=180, insecure=True, token=None):
    parsed = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    kwargs = {}
    if parsed.scheme == 'https' and insecure:
        kwargs['context'] = ssl._create_unverified_context()
    conn = cls(parsed.hostname, parsed.port, timeout=timeout, **kwargs)
    headers = {}
    if payload is not None:
        body = json.dumps(payload, separators=(',', ':'))
        headers['Content-Type'] = 'application/json'
    else:
        body = None
    if token:
        headers['Authorization'] = 'Bearer ' + token
    path = parsed.path or '/'
    if parsed.query:
        path += '?' + parsed.query
    conn.request('POST' if body else 'GET', path, body, headers)
    response = conn.getresponse()
    data = response.read()
    result = {'status': response.status, 'data': json.loads(data)}
    conn.close()
    if response.status != 200:
        raise RuntimeError({'non200': result})
    return result


def run_acceptance(url, output, token=None):
    models = request(url + '/models', token=token)
    ids = {entry['id'] for entry in models['data']['data']}
    assert 'M2' in ids, ids
    # The native EOS direct check needs low reasoning. Natural M2 reasoning
    # can exhaust 600 tokens on this trivial prompt and end finish=length;
    # that must not be misdiagnosed as identity/EOS/runtime regression.
    default = {'messages': [{'role': 'user', 'content': 'Reply only: M2 works'}],
               'model': 'M2', 'max_tokens': 600, 'temperature': 0,
               'chat_template_kwargs': {'reasoning_effort': 'low'}}
    chat = request(url + '/chat/completions', default, token=token)
    choice = chat['data']['choices'][0]
    assert choice['finish_reason'] == 'stop', choice
    assert 'M2 works' in (choice['message'].get('content') or ''), choice
    # Content is parsed separately from the optional reasoning channel. Do not
    # use the contradictory `.split('M2 works')[-1] == ''` assertion: it accepts
    # content-only reasoning and fails valid reasoning-containing low-effort replies.
    assert not (choice['message'].get('content') or '').lstrip().startswith('Here\'s a thinking process'), choice
    tool_payload = {'model': 'M2', 'max_tokens': 700, 'temperature': 0,
        'messages': [{'role': 'user', 'content': 'Call get_weather for Abu Dhabi directly. '
                'Output actual function invocation, not surrounding discussion.'}],
        'tool_choice': 'auto', 'tools': [{'type': 'function', 'function': {
            'name': 'get_weather', 'description': 'Current weather for a city',
            'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}},
            'required': ['location']}}}],
        'chat_template_kwargs': {'reasoning_effort': 'low'}}
    tool = request(url + '/chat/completions', tool_payload, timeout=300, token=token)
    tool_choice = tool['data']['choices'][0]
    assert tool_choice['finish_reason'] == 'tool_calls'
    tool_call = tool_choice['message']['tool_calls'][0]['function']
    assert tool_call['name'] == 'get_weather' and json.loads(tool_call['arguments'])['location'] == 'Abu Dhabi', tool_call
    return {'models': models, 'chat': chat, 'tool': tool,
            'chat_eos_native_no_stop_override': True}


def run_burst(url, count=8, token=None):
    payload = {'model': 'M2', 'max_tokens': 256, 'temperature': 0,
        'messages': [{'role': 'user', 'content': 'Write three concise bullet points explaining how expert parallelism improves mixture-of-experts inference.'}]}
    def timed_request(p):
        start = time.monotonic()
        response = request(url + '/chat/completions', p, token=token)
        return {'elapsed_s': time.monotonic() - start, 'response': response}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        rows = list(pool.map(timed_request, [dict(payload) for _ in range(count)]))
    elapsed = time.monotonic()-start
    tokens = [r['response']['data']['usage']['completion_tokens'] for r in rows]
    assert all(t == 256 for t in tokens), tokens
    return {'concurrency': count, 'token_total': sum(tokens), 'elapsed_s': elapsed,
            'prompt': payload['messages'][0]['content'],
            'per_request': rows,
            'aggregate_tokens_per_second': sum(tokens)/elapsed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--burst-count', type=int, default=0, choices=range(1,65),
        help='opt-in identical warm 256-token burst after checks; 1..64')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    url = valid_url(args.url)
    token = os.environ.get('M2_GATEWAY_TOKEN')
    result = run_acceptance(url, args.output, token)
    if args.burst_count:
        result['burst'] = run_burst(url, args.burst_count, token)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('ACCEPTANCE_EVIDENCE_WRITTEN', args.output)
    if 'burst' in result:
        print('BURST',result['burst'])
