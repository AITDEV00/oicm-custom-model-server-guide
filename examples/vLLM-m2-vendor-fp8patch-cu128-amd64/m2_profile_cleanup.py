"""Remove only temporary profiling envs from the profiled tuned M2 deployment.

After bounded trace export + analysis, restore the pre-profile tuned template.
Require deployment UID/generation14, live Ready pod/main restarts zero and exact
expected profile envs. Fail closed if any unrelated transformation happened.

A needlessly unexported profile may crash/pressure workers. Do not use this tool
as a generic rollout engine. Never replay m2_activate_maintenance solely for this.
"""
from __future__ import annotations
import copy
import json
import pathlib
import subprocess
import sys
import m2_profile_activation as act


def main():
    base = pathlib.Path(__file__).parent/'h200-moe-measurements'
    report = base/'trace-export/report.json'
    assert report.is_file()
    evidence = json.loads(report.read_text())
    assert len(evidence['traces']) == 8
    assert all(p['size'] > 0 for p in evidence['traces'])
    current = json.loads(act.relay('get', 'deployment', act.DEPLOY, '-o', 'json'))
    assert current['metadata']['uid'] == act.UID
    assert current['metadata']['generation'] == 14
    assert current['spec']['replicas'] == 1
    original = json.loads((base/'pre-profile-tuned-gen13.json').read_text())
    allowed = copy.deepcopy(original['spec']['template'])
    a = next(c for c in allowed['spec']['containers'] if c['name'] == 'main')
    a['env'] += act.ENV
    assert current['spec']['template'] == allowed
    patch = {'spec': {'replicas': 1,
        'template': copy.deepcopy(original['spec']['template'])}}
    (base/'profile-cleanup-plan.json').write_text(json.dumps(patch, indent=2)+'\n')
    fresh = json.loads(act.relay('get', 'deployment', act.DEPLOY, '-o', 'json'))
    assert fresh['metadata']['resourceVersion'] == current['metadata']['resourceVersion']
    assert fresh['spec']['template'] == allowed
    pods = json.loads(act.relay('get', 'pods', '-l', 'io.kompose.service='+act.DEPLOY, '-o', 'json'))['items']
    assert len(pods) == 1
    assert pods[0]['status']['phase'] == 'Running'
    state = next(s for s in pods[0]['status']['containerStatuses'] if s['name'] == 'main')
    assert state['ready'] and state['restartCount'] == 0
    relay_output = act.relay('patch', 'deployment', act.DEPLOY,
        '--type=strategic', '-p', json.dumps(patch, separators=(',', ':')))
    print('TRACE_EXPORTED_VERIFIED', len(evidence['traces']))
    print('REMOVAL_ONLY_PROFILE_ENVS', relay_output.strip())
    print('THIS_IS_NEXT_NORMAL_WORKER_RELOAD_NOT_EP_AB_ORIGINAL_TUNED_CONFIG_RETAINED')


if __name__ == '__main__':
    main()
