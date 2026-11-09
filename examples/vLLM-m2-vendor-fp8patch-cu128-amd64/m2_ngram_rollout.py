"""UID-matched, additive-only n-gram experiment for the healthy EP8 M2 worker.

Usage:
  python3 m2_ngram_rollout.py --apply-k 1 --confirm-deployment <exact name>
  python3 m2_ngram_rollout.py --apply-k 2 --confirm-deployment <exact name>
  python3 m2_ngram_rollout.py --keep-k 1 --confirm-deployment <exact name>

A measured k1 may be bounded-tested at k2, and the fastest measured k stays.
Never use this to restore no-spec: the user explicitly canceled that rollback.
Returns to the winning candidate use its exact measured template and have no
implicit no-spec intermediate. EP8 weights/context/EOS/pinned tile stay exact.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
EVIDENCE = ROOT / 'h200-moe-measurements' / 'ngram-20261110'
BASELINE = EVIDENCE / 'baseline-deployment.json'
BASELINE_PODS = EVIDENCE / 'baseline-pods.json'
DEPLOY = 'j-7ec325bf-0033-4a97-a688-e8ac30334349'
UID = 'cbe870c3-263f-4460-a6ae-05b9d4fb9b76'
ORIGINAL_POD_UID = '3a144e00-9115-45b1-a558-d97ded26884e'
IMAGE_DIGEST = 'sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9'


def relay(*args):
    command = ['kubectl', '-n', 'oik8s-cilium-system', 'exec', 'ad-debug', '--',
        'chroot', '/host', 'env', 'KUBECONFIG=/tmp/ad-kubeconfig', 'kubectl',
        '-n', 'adeo', *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed {result.returncode}: {result.stderr[-1000:]}')
    return result.stdout


def config(k):
    return json.dumps({'method': 'ngram', 'num_speculative_tokens': k,
        'prompt_lookup_min': 3, 'prompt_lookup_max': 5}, separators=(',', ':'))


def template_k(template):
    main = next(c for c in template['spec']['containers'] if c['name'] == 'main')
    args = main['args']
    assert args.count('--speculative-config') <= 1
    return 0 if not args.count('--speculative-config') else json.loads(
        args[args.index('--speculative-config') + 1])['num_speculative_tokens']


def normalized_equal(actual, baseline, remove_speculation):
    normalize, compare = copy.deepcopy(actual), copy.deepcopy(baseline)
    if remove_speculation:
        a = next(c for c in normalize['spec']['containers'] if c['name'] == 'main')['args']
        idx = a.index('--speculative-config')
        del a[idx:idx + 2]
    assert '--speculative-config' not in next(
        c for c in compare['spec']['containers'] if c['name'] == 'main')['args']
    assert normalize == compare, 'Unexpected non-speculative deployment delta'


def readiness(dep, pods):
    assert dep['metadata']['uid'] == UID
    assert dep['spec']['replicas'] == 1
    assert dep['spec']['strategy']['type'] == 'Recreate'
    assert dep['status']['observedGeneration'] == dep['metadata']['generation']
    assert dep['status'].get('readyReplicas') == 1
    assert dep['status'].get('availableReplicas') == 1
    assert len(pods['items']) == 1
    pod = pods['items'][0]
    assert pod['status']['phase'] == 'Running'
    status = next(s for s in pod['status']['containerStatuses'] if s['name'] == 'main')
    assert status['ready'] and status['restartCount'] == 0
    main = next(c for c in pod['spec']['containers'] if c['name'] == 'main')
    assert main['args'] == next(c for c in dep['spec']['template']['spec']['containers']
        if c['name'] == 'main')['args']
    return pod


def write_phase(path, dep, pods):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'deployment': dep, 'pods': pods}, indent=2) + '\n')


def main():

    os.environ.setdefault('KUBECONFIG', str(pathlib.Path.home() / '.kube/alain-oicm.conf'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-deployment', required=True)
    parser.add_argument('--apply-k', type=int, choices=[1, 2])
    parser.add_argument('--keep-k', type=int, choices=[1, 2],
        help='keep/retain tested winner from an earlier measured candidate phase')
    parser.add_argument('--capture', type=pathlib.Path,
        help='read-only current identity+runtime snapshot; does not set phase')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--record-tested', type=pathlib.Path,
        help='commit phase ONLY after the named valid measurement file exists')
    options = parser.parse_args()
    if options.confirm_deployment != DEPLOY:
        raise SystemExit('--confirm-deployment must exactly match the target deployment')
    actions = [bool(options.apply_k), bool(options.keep_k), bool(options.capture),
        bool(options.record_tested)]
    if sum(actions) != 1:
        raise SystemExit('select exactly one apply, keep-winner, capture, or tested action')
    baseline_raw = json.loads(BASELINE.read_text())
    assert baseline_raw['metadata']['resourceVersion'] == '470304584'
    assert baseline_raw['metadata']['generation'] == 17
    assert BASELINE.is_file()
    baseline_pods = json.loads(BASELINE_PODS.read_text())
    original = readiness(baseline_raw, baseline_pods)
    assert original['metadata']['uid'] == ORIGINAL_POD_UID
    assert '-84fc97ff-' in original['metadata']['name']
    baseline = baseline_raw['spec']['template']
    assert template_k(baseline) == 0
    main_c = next(c for c in baseline['spec']['containers'] if c['name'] == 'main')
    args = main_c['args']
    assert main_c['image'].endswith(IMAGE_DIGEST)
    assert '--enable-expert-parallel' in args
    assert args[args.index('--tensor-parallel-size') + 1] == '8'
    assert args[args.index('--max-model-len') + 1] == '262144'
    assert args[args.index('--kv-cache-dtype') + 1] == 'fp8_e4m3'
    assert json.loads(args[args.index('--override-generation-config') + 1]) == {'eos_token_id': 250019}
    assert '--compilation-config' not in args
    assert next(e for e in main_c['env'] if e['name'] == 'VLLM_TUNED_CONFIG_FOLDER')['value'] == '/etc/m2-moe-tuned'
    assert not any(e['name'].startswith('VLLM_PROFILER') or e['name'] == 'VLLM_TORCH_PROFILER_DIR'
        for e in main_c['env'])

    live = json.loads(relay('get', 'deployment', DEPLOY, '-o', 'json'))
    pods = json.loads(relay('get', 'pods', '-l', 'io.kompose.service=' + DEPLOY, '-o', 'json'))
    if options.capture:
        write_phase(options.capture, live, pods)
        print('READ_ONLY_SNAPSHOT', options.capture,
            'GENERATION', live['metadata'].get('generation'), 'K', template_k(live['spec']['template']))
        return
    pod = readiness(live, pods)
    current_k = template_k(live['spec']['template'])
    current_path = EVIDENCE / f'current-generation{live["metadata"]["generation"]}.json'
    write_phase(current_path, live, pods)
    if options.record_tested:
        report = json.loads(options.record_tested.read_text())
        assert report['pod_uid'] == pod['metadata']['uid']
        assert report['ngram_k'] == current_k
        assert report['all_checks_passed'] is True
        state = EVIDENCE / f'k{current_k}-tested.json'
        state.write_text(json.dumps({'deployment_uid': UID, 'pod_uid': pod['metadata']['uid'],
            'ngram_k': current_k, 'generation': live['metadata']['generation'],
            'measurement': options.record_tested.name}, indent=2) + '\n')
        print('TESTED_PHASE_COMMITTED', 'ngram_k', current_k,
            'generation', live['metadata']['generation'])
        return

    desired_k = options.keep_k or options.apply_k
    if current_k == desired_k:
        assert (EVIDENCE / f'k{current_k}-tested.json').is_file()
        print('ALREADY_CURRENT_TESTED_CONFIG_RETAINED', desired_k, 'pod', pod['metadata']['name'])
        return
    if options.apply_k == 1:
        assert (live['metadata']['generation'], current_k) == (17, 0), 'k1 must start from original healthy gen17'
    elif options.apply_k == 2:
        state = json.loads((EVIDENCE / 'k1-tested.json').read_text())
        assert state['pod_uid'] == pod['metadata']['uid']
        assert (live['metadata']['generation'], current_k) == (18, 1)
    else:
        assert (live['metadata']['generation'], current_k) == (19, 2)
        assert (EVIDENCE / 'k2-tested.json').is_file()
        result = json.loads((EVIDENCE / f'k{current_k}-observed.json').read_text())
        assert result['ngram_k'] == current_k and result['pod_uid'] == pod['metadata']['uid']
        assert result['all_checks_passed'] is True
    # No baseline restoration: copy the healthy CURRENT M2 args, not no-spec baseline's alias.
    desired = copy.deepcopy(live)
    desired['metadata']['resourceVersion'] = None
    args = copy.deepcopy(next(c for c in live['spec']['template']['spec']['containers']
        if c['name'] == 'main')['args'])
    desired_args = next(c for c in desired['spec']['template']['spec']['containers']
        if c['name'] == 'main')['args']
    if current_k == 0:
        desired_args += ['--speculative-config', config(desired_k)]
    else:
        desired_args[desired_args.index('--speculative-config') + 1] = config(desired_k)
    assert template_k(desired['spec']['template']) == desired_k
    normalized_equal(desired['spec']['template'], baseline, True)
    if current_k:
        option_index = args.index('--speculative-config')
        assert desired_args.index('--speculative-config') == option_index
        difference = [(i, a, b) for i, (a, b) in enumerate(zip(desired_args, args)) if a != b]
        assert difference == [(option_index + 1, config(desired_k), args[option_index + 1])]
        assert len(desired_args) == len(args)
        assert args[option_index + 1] == config(current_k)
    else:
        assert desired_args[:-2] == args
    plan = {'allowed_delta': f'ngram k{current_k} -> k{desired_k}; ONLY arg list value, no no-spec stage',
        'expected_uid': UID, 'current_resource_version': live['metadata']['resourceVersion'],
        'current_generation': live['metadata']['generation'], 'current_pod_uid': pod['metadata']['uid'],
        'next_template': desired, 'image_digests_untouched': True,
        'weights_pvc_hf_overrides_tuned_kernel_context_eos_and_controls_unchanged': True}
    for filename, previous in {'current_resource_version': live['metadata']['resourceVersion'],
            'current_generation': live['metadata']['generation'],
            'current_pod_uid': pod['metadata']['uid']}.items():
        fresh = json.loads(relay('get', 'deployment', DEPLOY, '-o', 'json'))
        if filename == 'current_resource_version':
            assert fresh['metadata']['resourceVersion'] == previous
        elif filename == 'current_generation':
            assert fresh['metadata']['generation'] == previous
        else:
            assert fresh['metadata']['uid'] == UID
            fresh_pods = json.loads(relay('get', 'pods', '-l', 'io.kompose.service=' + DEPLOY, '-o', 'json'))
            assert readiness(fresh, fresh_pods)['metadata']['uid'] == previous
    if options.keep_k:
        # Only the single main n-gram config argument changes. Use a scoped
        # args-list patch, not a whole-template or no-spec restore operation.
        patch = {'spec': {'template': {'spec': {'containers': [
            {'name': 'main', 'args': desired_args}]}}}}
    else:
        patch = {'metadata': {'resourceVersion': live['metadata']['resourceVersion']},
            'spec': {'template': desired}}
    plan['deployment_patch'] = patch
    output = EVIDENCE / f'phase-k{desired_k}-plan.json'
    output.write_text(json.dumps(plan, indent=2) + '\n')
    relay('patch', 'deployment', DEPLOY, '--type=strategic', '-p',
        json.dumps(patch, separators=(',', ':')))
    print('APPLIED_NGRAM_EXPERIMENT_VARIANT', desired_k, 'ONLY_SPECULATIVE_ARG_DELTA', output)
    print('NEXT: wait full fresh-pod model readiness; DO NOT treat public gate readiness as M2 ready')


if __name__ == '__main__':
    main()
