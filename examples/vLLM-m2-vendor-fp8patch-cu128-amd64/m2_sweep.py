"""Independent single-H200, model-free Triton block FP8 candidate sweep."""
from __future__ import annotations
# Canonical coordinator for this bounded representative subset.
# m2_moe_run/m2_moe_worker experiments are not operational entrypoints.
import argparse, itertools, json, os, pathlib, signal, subprocess, sys, time
import m2_moe_core as c

ROOT = pathlib.Path('/work/m2-tuner-work/m2-tuned-moe')
ENV = os.environ.copy()
ENV.update(PYTHONPATH='/material:/opt/vllm', PYTHONUNBUFFERED='1',
           M2_TUNING_TRIAL_GPU_COUNT='1')


def guard():
    assert os.environ['M2_TUNING_CONFIRM'] == 'standalone-offline'
    assert os.environ['M2_TUNING_TARGET_UID'] == c.TARGET_UID
    assert os.environ.get('M2_NO_MODEL_DOWNLOAD') == '1'
    assert os.environ.get('M2_NO_SERVICE_PATCH') == '1'
    assert os.environ.get('M2_MAINTENANCE_CONFIRM') == 'spare-h200-or-m2-stopped'
    assert c.benchmark_source_hash() == c.BENCHMARK_SHA256
    assert not pathlib.Path('/pvc-home').exists()


def candidates():
    return list(itertools.product((16, 64, 256), (128, 256),
                (1, 32), (4, 8), (3, 5)))


def call(m, config, numeric, iters=20, timeout=150):
    cmd = [sys.executable, '/work/m2_tile_worker.py', '--m', str(m),
           '--config', json.dumps(config), '--seed', '0', '--iters', str(iters)]
    if numeric:
        cmd.append('--numeric')
    process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=ENV, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError(f'Trial timed out M{m}: {config}') from exc
    begin = 'M2_TRIAL_RESULT='
    rows = [json.loads(line[len(begin):]) for line in stdout.splitlines()
            if line.startswith(begin)]
    if process.returncode or len(rows) != 1:
        return {'M': m, 'config': config, 'status': 'error',
                'rc': process.returncode, 'stdout': stdout[-8000:],
                'stderr': stderr[-8000:]}
    return rows[0]


def main(args):
    guard()
    ROOT.mkdir(parents=True, exist_ok=True)
    cp = ROOT / 'checkpoint.json'
    if cp.exists():
        state = json.loads(cp.read_text())
        assert state['sizes'] == [1, 2, 4, 8]
        assert state['shape'] == c.SHAPE
        assert state['benchmark_sha256'] == c.BENCHMARK_SHA256
        state['trials'] = {key: row for key, row in state['trials'].items()
                           if row.get('time_us', 0) > 0 and row.get('numeric_pass') is True
                           or row.get('numeric_pass') is None and row.get('time_us', 0) > 0}
    else:
        state = {'schema': 'm2.moe.synthetic.sweep.v1', 'shape': c.SHAPE,
                 'sizes': [1, 2, 4, 8], 'benchmark_sha256': c.BENCHMARK_SHA256,
                 'scope': '48 representative tiles per M + default, not exhaustive',
                 'trials': {}, 'errors': [], 'selected': {}, 'ab': {}}
    if args.numbers_only:
        print('EXACT_BUDGET_PER_M_INCLUDING_DEFAULT', len(candidates()) + 1)
        return len(candidates()) + 1
    for m in state['sizes']:
        start = time.monotonic()
        combinations = [(16, 128, 32, 4, 3)] + list(candidates())
        # A resumed run skips only already validated rows, not failures.
        # 40 valid +8 over-memory exclusions per size is the expected result.
        for index, (bm, bn, gm, warps, stages) in enumerate(combinations):
            key = f'{m}:{bm}:{bn}:{gm}:{warps}:{stages}'
            if key in state['trials']:
                print('RESUME', key, flush=True)
                continue
            cfg = {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn, 'BLOCK_SIZE_K': 128,
                   'GROUP_SIZE_M': gm, 'num_warps': warps, 'num_stages': stages}
            try:
                row = call(m, cfg, True, timeout=args.timeout)
                if 'time_us' in row and row['numeric_pass'] is True:
                    state['trials'][key] = row
                else:
                    state['errors'].append(row)
                elapsed = time.monotonic() - start
                print('RESULT', key, row.get('time_us'), row.get('numeric_pass'),
                      row.get('error'), 'count', len(state['trials']), 'elapsed_total', round(elapsed, 1), flush=True)
            except Exception as exc:
                state['errors'].append({'M': m, 'config': cfg, 'error': str(exc)})
            atomic(cp, state)
    shortlist = {}
    for m in state['sizes']:
        rows = [r for r in state['trials'].values() if r['M'] == m and r.get('numeric_pass') is True
                and r['config'] != c.DEFAULT]
        assert rows, f'No validated candidate M{m}'
        choice = min(rows, key=lambda r: r['time_us'])
        pairs = []
        for round_no in range(3):
            for label, cfg in [('candidate', choice['config']), ('default', c.DEFAULT)]:
                r = call(m, cfg, True, iters=128, timeout=args.timeout)
                if not r.get('numeric_pass'):
                    raise RuntimeError(f'AB failed: {r}')
                pairs.append({'round': round_no, 'label': label, **r})
                atomic(ROOT / 'finalist-ab.json', pairs)
        candidate_us = [r['time_us'] for r in pairs if r['label'] == 'candidate']
        default_us = [r['time_us'] for r in pairs if r['label'] == 'default']
        med = lambda xs: sorted(xs)[len(xs)//2]
        ab = {'candidate': choice['config'],
              'default_median_us': med(default_us),
              'candidate_median_us': med(candidate_us),
              'speedup_local_expert': med(default_us) / med(candidate_us),
              'rounds': pairs, 'numerical_pass': True}
        print('FINALIST_AB', m, ab['speedup_local_expert'], flush=True)
        shortlist[m] = ab
        c.write_json(ROOT / f'finalist-ab-m{m}.json', ab)
    c.write_json(ROOT / 'finalist-ab.json', shortlist)
    assert all(shortlist[m]['speedup_local_expert'] > 1 for m in state['sizes'])
    native = {'triton_version': '3.5.0', **{str(m): shortlist[m]['candidate'] for m in state['sizes']}}
    native_path = ROOT / c.ARTIFACT_NAME
    c.write_json(native_path, native)
    manifest = {'scope': 'solid vendor synthetic local-expert timing, not full M2 throughput',
                'search': '49 per M (including default), non-exhaustive subset of effective 320',
                'attempt_count': len(state['trials']) + len(state['errors']),
                'trials': state['trials'], 'errors': state['errors'],
                'selected_ab': shortlist, 'source_benchmark_sha256': c.BENCHMARK_SHA256}
    c.write_json(ROOT / 'evidence.json', manifest)
    report = c.validate_artifact(native_path, ROOT)
    c.write_json(ROOT / 'native-loader-verification.json', report)
    print('FULL_VALIDATED_NATIVE_ARTIFACT', native_path, flush=True)
    print('VERBOSE_EVIDENCE', ROOT / 'evidence.json', flush=True)


def atomic(path, value):
    c.write_json(path, value)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--numbers-only', action='store_true')
    p.add_argument('--max-time', type=int, default=2400)
    args = p.parse_args()
    main(args)
