"""One UID-matched M2 restore from the saved pre-maintenance Deployment.

No automatic rollout: --apply requires an exact deployment-name confirmation.
Original PVC/model/init/license/source/runtime arguments stay untouched.
The only optional delta is the separately native-loader-validated tile file.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, subprocess
import m2_restore

def base_ad(stdin=None):
    args = ['kubectl', '-n', 'oik8s-cilium-system', 'exec']
    if stdin is not None:
        args.append('-i')
    args += ['ad-debug', '--', 'chroot', '/host', 'env',
             'KUBECONFIG=/tmp/ad-kubeconfig', 'kubectl']
    if stdin is not None:
        args.append('-i')
    return args + ['-n', 'adeo']


def relay(args, *, stdin=None):
    command = base_ad(stdin) + args
    result = subprocess.run(command, check=False, input=stdin,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'AD relay failed ({result.returncode}): {result.stderr[-4000:]}')
    return result


def delta_guard(original_path, tuned_path, plan):
    """Prove template change is only replicas + native CM/env/read-only mount."""
    original = json.loads(pathlib.Path(original_path).read_text())
    expected = copy.deepcopy(original['spec']['template'])
    actual = plan['deployment_patch']['spec']['template']
    assert actual['metadata'] == expected['metadata']
    actual_spec = actual['spec']
    expected_spec = expected['spec']
    if not tuned_path:
        assert actual == expected
        assert plan['artifact_configmap_create'] is None
        return
    original_main = next(c for c in expected_spec['containers'] if c['name'] == 'main')
    actual_main = next(c for c in actual_spec['containers'] if c['name'] == 'main')
    expected_mounts = original_main['volumeMounts']
    actual_mounts = actual_main['volumeMounts']
    assert actual_mounts[:-1] == expected_mounts
    assert actual_mounts[-1] == {'name': 'm2-moe-tuned-config',
        'mountPath': m2_restore.FOLDER, 'readOnly': True}
    assert actual_main.get('env', [])[:-1] == original_main.get('env', [])
    assert actual_main['env'][-1] == {'name': 'VLLM_TUNED_CONFIG_FOLDER', 'value': m2_restore.FOLDER}
    new_volume = {'name': 'm2-moe-tuned-config', 'configMap': {'name': m2_restore.CM,
        'items': [{'key': 'native.json', 'path': m2_restore.ARTIFACT}]}}
    assert actual_spec['volumes'][:-1] == expected_spec['volumes']
    assert actual_spec['volumes'][-1] == new_volume
    for ac, oc in zip(actual_spec['containers'], expected_spec['containers']):
        ac = copy.deepcopy(ac); oc = copy.deepcopy(oc)
        ac.pop('env', None); ac.pop('volumeMounts', None)
        oc.pop('env', None); oc.pop('volumeMounts', None)
        assert ac == oc
    ac = copy.deepcopy(actual_spec); ec = copy.deepcopy(expected_spec)
    ac['containers'] = ec['containers']
    ac['volumes'] = ec['volumes']
    assert ac == ec
    artifact = json.loads(plan['artifact_configmap_create']['data']['native.json'])
    native_file = pathlib.Path(tuned_path)
    assert native_file.name == m2_restore.ARTIFACT
    assert native_file.stat().st_size > 0
    assert json.loads(native_file.read_text()) == artifact
    digest = hashlib.sha256(native_file.read_bytes()).hexdigest()
    report_path = native_file.parent / 'native-loader-verification.json'
    report = json.loads(report_path.read_text())
    assert report['validated_vendor_loader'] is True
    assert report['sha256'] == digest
    assert report['loaded_contents'] == artifact
    print('FULL_ORIGINAL_TEMPLATE_PRESERVED_EXCEPT_NATIVE_CONFIG', flush=True)


def prepare(original, current, native=None, save=None):
    cap = pathlib.Path(current)
    dep = json.loads(relay(['get', 'deployment', m2_restore.DEPLOY, '-o', 'json']).stdout)
    cap.parent.mkdir(parents=True, exist_ok=True)
    cap.write_text(json.dumps(dep, sort_keys=True, indent=2)+'\n')
    snapshot = json.loads(cap.read_text())
    assert snapshot['spec']['replicas'] == 0
    assert snapshot['metadata']['uid'] == m2_restore.UID
    assert cap.stat().st_size > 1000
    plan = m2_restore.main(original, native, current)
    delta_guard(original, native, plan)
    if save:
        pathlib.Path(save).write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def apply(plan, current, original, tuned):
    delta_guard(original, tuned, plan)
    dep = json.loads(relay(['get', 'deployment', m2_restore.DEPLOY,
                            '-o', 'json']).stdout)
    assert dep['metadata']['uid'] == m2_restore.UID
    assert dep['metadata']['resourceVersion'] == json.loads(pathlib.Path(current).read_text())['metadata']['resourceVersion']
    assert dep['spec']['replicas'] == 0
    assert not dep.get('status', {}).get('replicas', 0)
    # Reject changing model/PVC fields already validated in the snapshot plan.
    template = plan['deployment_patch']['spec']['template']
    assert any(c['name']=='init-model-downloader' for c in template['spec']['initContainers'])
    if plan['artifact_configmap_create']:
        # AD proxy kubectl cannot accept an inner -i before the plugin. One
        # untrusted-but-literal sh string avoids that wrapper while forwarding
        # only the native artifact JSON, not model/PVC/license/source payloads.
        result = subprocess.run(['kubectl', '-n', 'oik8s-cilium-system', 'exec',
            '-i', 'ad-debug', '--', 'sh', '-c',
            "chroot /host sh -c 'env KUBECONFIG=/tmp/ad-kubeconfig kubectl -n adeo apply -f -'"],
            input=json.dumps(plan['artifact_configmap_create']), check=True,
            capture_output=True, text=True)
        print('VALIDATED_NATIVE_CONFIGMAP_APPLIED', result.stdout.strip(), flush=True)
    relay(['patch', 'deployment', m2_restore.DEPLOY, '--type=strategic',
           '-p', json.dumps(plan['deployment_patch'], separators=(',', ':'))])
    return {'restore_applied_once': True, 'uid': m2_restore.UID,
            'native_tuned_config': bool(plan['artifact_configmap_create'])}


if __name__ == '__main__':
    os.environ.setdefault('KUBECONFIG', str(pathlib.Path.home()/'.kube/alain-oicm.conf'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    parser.add_argument('--tuned-native')
    parser.add_argument('--current', required=True, help='fresh snapshot path')
    parser.add_argument('--save-plan', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--confirm-deployment')
    options = parser.parse_args()
    plan = prepare(options.original, options.current, options.tuned_native,
                   options.save_plan)
    if options.apply:
        if options.confirm_deployment != m2_restore.DEPLOY:
            raise SystemExit('--apply requires exact --confirm-deployment')
        print(json.dumps(apply(plan, options.current, options.original,
                               options.tuned_native)), flush=True)
    else:
        print('READ_ONLY_PLAN_PREPARED_NOT_APPLIED', options.save_plan)
