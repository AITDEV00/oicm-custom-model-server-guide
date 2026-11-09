"""Read-only matching retained live n-gram capture wrapper.

This host script NEVER patches or restarts the Deployment. It must run on the
already observed ready Generation20 (retained K1) pod of the requested
configuration. The exact homogeneous local source is streamed into the worker
with raw-byte SHA256 verified before an in-process exec; unlike the historical
old-helper mode this never imports chroot/container's existing util filenames.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
EVIDENCE = ROOT / 'h200-moe-measurements' / 'ngram-20261110'
DEPLOY = 'j-7ec325bf-0033-4a97-a688-e8ac30334349'
UID = 'cbe870c3-263f-4460-a6ae-05b9d4fb9b76'
DIRECT_NODE = 'prd-infr-k8h200'
BASELINE = EVIDENCE / 'baseline-deployment.json'


def krelay(*args, stdin=False):
    return ['kubectl', '-n', 'oik8s-cilium-system', 'exec', *(["-i"] if stdin else []),
        'ad-debug', '--', 'chroot', '/host', 'env', 'KUBECONFIG=/tmp/ad-kubeconfig',
        'kubectl', '-n', 'adeo', *args]


def call(*args, **kwargs):
    result = subprocess.run(krelay(*args, **kwargs), capture_output=True, text=True,
        timeout=kwargs.pop('timeout', 30))
    if result.returncode:
        raise RuntimeError({'exit': result.returncode, 'stderr': result.stderr[-2000:]})
    return result.stdout


def phase_pod(k):
    result = json.loads(call('get', 'pods', '-l', 'io.kompose.service=' + DEPLOY, '-o', 'json'))
    if len(result['items']) != 1:
        raise AssertionError('exactly one service pod required')
    pod = result['items'][0]
    assert pod['metadata']['namespace'] == 'adeo'
    assert pod['metadata']['ownerReferences'][0]['kind'] == 'ReplicaSet'
    assert pod['spec'].get('nodeName') == DIRECT_NODE
    status = next(x for x in pod['status']['containerStatuses'] if x['name'] == 'main')
    assert status['ready'] and status['restartCount'] == 0
    assert not pod['metadata'].get('deletionTimestamp')
    main = next(c for c in pod['spec']['containers'] if c['name'] == 'main')
    args = main['args']
    assert '--enable-expert-parallel' in args
    assert args[args.index('--tensor-parallel-size') + 1] == '8'
    assert '--VLLM_TORCH_PROFILER_DIR' not in args
    if k == 0:
        assert '--speculative-config' not in args
    else:
        cfg = json.loads(args[args.index('--speculative-config') + 1])
        assert cfg == {'method': 'ngram', 'num_speculative_tokens': k,
            'prompt_lookup_min': 3, 'prompt_lookup_max': 5}
    dep = json.loads(call('get', 'deployment', DEPLOY, '-o', 'json'))
    assert dep['metadata']['uid'] == UID
    assert dep['metadata']['generation'] == dep['status']['observedGeneration']
    assert dep['status']['readyReplicas'] == 1
    # Deployment PodSpec and the created pod differ in runtime-populated token
    # projection and possibly CNI annotations, so require exact necessary
    # container execution contract fields rather than a false whole-object
    # equality wrapped in an always-true assertion.
    template_main = next(c for c in dep['spec']['template']['spec']['containers'] if c['name'] == 'main')
    assert template_main['args'] == main['args']
    assert template_main['image'] == main['image']
    (EVIDENCE / f'k{k}-live-pod.json').write_text(json.dumps(pod, indent=2) + '\n')
    (EVIDENCE / f'k{k}-live-deployment.json').write_text(json.dumps(dep, indent=2) + '\n')
    return pod


def remote(*args, stdin='', timeout=300):
    command = krelay('exec', '-i', args[0], '-c', 'main', '--', 'python3', '-', *args[1:], stdin=True)
    # Explicit shell shape avoids interpreting wrapper bytes as service code.
    result = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError({'remote_exit': result.returncode,
            'stderr_tail': result.stderr[-4000:], 'stdout_tail': result.stdout[-500:]})
    return result.stdout


def main():
    os.environ.setdefault('KUBECONFIG', str(pathlib.Path.home() / '.kube/alain-oicm.conf'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram-k', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--report-name', help='distinct filename suffix for reload confirmation')
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    phase_descriptor = json.loads(call('get','deployment',DEPLOY,'-o','json'))
    assert phase_descriptor['metadata']['generation'] == 20 and args.ngram_k == 1
    pod = phase_pod(args.ngram_k)
    name = pod['metadata']['name']
    uid = pod['metadata']['uid']
    data = (ROOT / 'm2_ngram_benchmark.py').read_bytes()
    expected = hashlib.sha256(data).hexdigest()
    arguments = ['--pod-uid', uid, '--ngram-k', str(args.ngram_k),
                 '--output', f'/tmp/m2-ngram-k{args.ngram_k}-observed.json',
                 '--with-repeat', '--full-two-eight']
    command = krelay('exec', '-i', name, '-c', 'main', '--', 'python3', '-c',
        "import sys,pathlib,hashlib,runpy;"
        "raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).digest()==bytes.fromhex(sys.argv[-1]);"
        "pathlib.Path('/tmp/m2-ngram-pod-uid').write_text(sys.argv[-2]);"
        "sys.argv=['m2_ngram_benchmark.py']+sys.argv[1:-2];"
        "exec(compile(raw,'m2_ngram_benchmark.py','exec'),{'__name__':'__main__'})",
        *arguments, uid, expected, stdin=True)
    # Independent raw-code stdin is not an argv source-file argument and cannot
    # be mistaken for the helper source passed to python -.
    result = subprocess.run(command, input=data, capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    if not result.stdout.strip():
        raise RuntimeError('benchmark returned no result')
    report_output = subprocess.run(krelay('exec', name, '-c', 'main', '--',
        'cat', f'/tmp/m2-ngram-k{args.ngram_k}-observed.json'), capture_output=True, timeout=30)
    if report_output.returncode:
        raise RuntimeError(report_output.stderr[-2000:])
    report = json.loads(report_output.stdout)
    assert report['pod_uid'] == uid and report['ngram_k'] == args.ngram_k
    assert report['all_checks_passed'] is True
    output = EVIDENCE / f'k{args.ngram_k}-observed.json'
    if args.report_name:
        output = EVIDENCE / f'k{args.ngram_k}-{args.report_name}.json'
    output.write_bytes(report_output.stdout)
    print('VERIFIED_STDIN_MODULE_SHA256', expected)
    print(result.stdout.decode())


if __name__ == '__main__':
    main()
