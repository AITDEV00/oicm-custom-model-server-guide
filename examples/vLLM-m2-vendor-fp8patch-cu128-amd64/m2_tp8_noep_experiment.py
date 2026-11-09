"""DOCUMENTATION ONLY: blocked TP8-without-EP M2 experiment.

On 2026-11-10 tested exact original M2 FP8 checkpoint image / loader. When EP
was removed at TP8, local intermediate width became 1792/8 = 224. Every TP
rank failed creating weights before checkpoint loading:
  ValueError: The output_size of gate's and up's weight = 224 is not
  divisible by weight quantization block_n = 128.
This is a hard FP8 shape-compatibility failure, NOT a performance result.
Service was restored to working TP8+EP8 tuned. The current n-gram choice is
preserved separately (no zero-spec phase by default).
Do not replay this script nor rename the E24,N1792 tuning file to E192,N224.
A future supported TP-only kernel must validate group/block-n compatibility
for original compressed-tensors W8A8 FP8 weights instead of altering weights.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pathlib,subprocess
raise RuntimeError('BLOCKED: TP8-only FP8 M2 fails output_size=224 vs block_n=128; this historical no-EP experiment had no valid throughput result.')
import m2_profile_activation as act

BASE=pathlib.Path(__file__).parent/'h200-moe-measurements'
PROFILE_NAMES={x['name']for x in act.ENV}

def load_current():
 return json.loads(act.relay('get','deployment',act.DEPLOY,'-o','json'))

def main_container(template):
 return next(c for c in template['spec']['containers']if c['name']=='main')

def invariant(template,require_ep):
 assert template['spec']['runtimeClassName']=='nvidia'
 main=main_container(template)
 assert 'm2-vllm@sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9' in main['image']
 assert main['resources']['limits']=={'cpu':'16','memory':'1Ti','nvidia.com/h200':'8'}
 assert main['args'].count('--enable-expert-parallel')==int(require_ep)
 assert main['args'][main['args'].index('--tensor-parallel-size')+1]=='8'
 assert main['args'][main['args'].index('--max-model-len')+1]=='262144'
 assert main['args'][main['args'].index('--override-generation-config')+1]=='{"eos_token_id":250019}'
 assert '--speculative-config'not in main['args']
 assert main['args'][0]=='/pvc-home/app/download/base_model'
 assert main['args'][main['args'].index('--kv-cache-dtype')+1]=='fp8_e4m3'
 assert main['args'][main['args'].index('--served-model-name')+1]=='M2'
 assert not any('VLLM_TUNED_CONFIG_FOLDER'==e['name'] and e['value']!='/etc/m2-moe-tuned'for e in main['env'])
 next(c for c in template['spec']['initContainers']if c['name']=='init-model-downloader')
 pvc=next(v for v in template['spec']['volumes']if v['name'].startswith('pvc-'))
 assert pvc['persistentVolumeClaim']['claimName']=='pvc-7ec325bf-0033-4a97-a688-e8ac30334349'
 assert not any(m['mountPath']=='/pvc-home' and m.get('subPath')for m in main['volumeMounts'])

def transforms(dep):
 invariant(dep['spec']['template'],True)
 template=copy.deepcopy(dep['spec']['template']);main=main_container(template)
 original_profile=[e for e in main['env']if e['name']in PROFILE_NAMES]
 assert original_profile==act.ENV,{'unexpected_profiler_envs':original_profile}
 main['args'].remove('--enable-expert-parallel')
 main['env']=[e for e in main['env']if e['name']not in PROFILE_NAMES]
 invariant(template,False)
 original_template=copy.deepcopy(dep['spec']['template'])
 # Simple, exact, no other container/env/arg/resource changes test.
 normalized=copy.deepcopy(template);m=main_container(normalized)
 m['args'].insert(m['args'].index('--tensor-parallel-size')+2,'--enable-expert-parallel')
 m['env']+=copy.deepcopy(act.ENV)
 assert normalized==original_template,'would change non-experiment fields'
 return template

def plan(dep):
 assert dep['metadata']['uid']==act.UID
 assert dep['metadata']['generation']==14
 assert dep['spec']['strategy']['type']=='Recreate'
 assert dep['spec']['replicas']==dep['status']['readyReplicas']==1
 assert len([x for x in dep.get('status',{}).get('conditions',[])if not x])==0
 assert dep['status'].get('unavailableReplicas',0)==0
 template=transforms(dep)
 before=json.dumps(dep['spec']['template'],sort_keys=True).encode()
 return {'source_generation':14,'source_resourceVersion':dep['metadata']['resourceVersion'],
  'source_template_sha256':hashlib.sha256(before).hexdigest(),
  'user_authorized_change':'TP8 without EP for performance-only A/B',
  'ngram_enabled':False,'new_weights_files':False,'new_runtime_code':False,
  'default_config_expected':'E=192,N=224,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128]',
  'deployment_patch':{'spec':{'replicas':1,'template':template}}}

def guarded_apply(p):
 assert os.environ.get('M2_FIRST_EXPERIMENT_CONFIRM')=='tp8-noep-only'
 assert p['validation_scope']=='session-authorized-one-relaunch'
 dep=load_current()
 assert dep['metadata']['uid']==act.UID
 assert dep['metadata']['resourceVersion']==p['source_resourceVersion']
 assert dep['metadata']['generation']==p['source_generation']
 assert dep['spec']==p['source_spec'],'source changed since plan; do not patch blindly'
 assert dep['spec']['replicas']==dep['status']['readyReplicas']==1
 assert dep['status'].get('unavailableReplicas',0)==0
 template=p['deployment_patch']['spec']['template']
 act.relay('patch','deployment',act.DEPLOY,'--type=strategic','-p',json.dumps(
     {'spec':{'template':template}},separators=(',',':')))
 print('APPLY_SENT_TP8_NO_EP_SINGLE_EXPERIMENT_NO_NGRAM')

if __name__=='__main__':
 os.environ.setdefault('KUBECONFIG',str(pathlib.Path.home()/'.kube/alain-oicm.conf'))
 parser=argparse.ArgumentParser()
 parser.add_argument('--apply',action='store_true')
 options=parser.parse_args()
 dep=load_current();p=plan(dep)
 p['validation_scope']='session-authorized-one-relaunch'
 p['source_spec']=dep['spec']
 (BASE/'tp8-noep-plan.json').write_text(json.dumps(p,indent=2)+'\n')
 (BASE/'pre-tp8-noep-generation14.json').write_text(json.dumps(dep,indent=2)+'\n')
 if options.apply:
  guarded_apply(p)
 else:
  print('PLAN_ONLY_VALIDATED_NO_APPLY',json.dumps({k:v for k,v in p.items()if k not in('deployment_patch','source_spec')},indent=2))
