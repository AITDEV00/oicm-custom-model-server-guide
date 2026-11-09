import json, os, pathlib, torch
from unittest.mock import patch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsW8A8Fp8MoEMethod

# Quantization scheme fields of this real checkpoint: 128x128 static weights,
# dynamic group128 activations. The tested slots and callbacks are produced
# by create_weights(), not a small artificial tensor-of-one approximation.
overrides = {'config_groups': {'group_0': {'targets': ['Linear'], 'format': 'float-quantized', 'input_activations': {'dynamic': True, 'num_bits': 8, 'type': 'float', 'symmetric': True, 'strategy': 'group', 'group_size': 128, 'actorder': None, 'block_structure': None, 'observer': None, 'observer_kwargs': {}}, 'output_activations': None, 'weights': {'dynamic': False, 'num_bits': 8, 'type': 'float', 'symmetric': True, 'strategy': 'block', 'group_size': None, 'block_structure': [128, 128], 'actorder': None, 'observer': 'memoryless_minmax', 'observer_kwargs': {}}}}}
config = overrides
scheme = CompressedTensorsConfig._quantization_scheme_map_from_config(config)['Linear']
weights = scheme['weights']; activations = scheme['input_activations']
assert weights.strategy == 'block'
assert weights.block_structure == [128, 128]
assert activations.strategy == 'group'
assert activations.group_size == 128
method = CompressedTensorsW8A8Fp8MoEMethod.__new__(CompressedTensorsW8A8Fp8MoEMethod)
method.weight_quant = weights
method.input_quant = activations
method.weight_block_size = weights.block_structure
method.block_quant = True
method.static_input_scales = False

class Experts(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.expert_map = {g: torch.tensor(g - owner*24 if owner*24 <= g < (owner+1)*24 else -1) for g in range(192)}
        self.ep_rank = owner
        self.tp_size = 1
        self.tp_rank = 0
        self.quant_config = None
        self.moe_config = type('Cfg', (), {'is_act_and_mul': True})()
        self.quant_method = method
        self._map_global_expert_id_to_local_expert_id = FusedMoE._map_global_expert_id_to_local_expert_id.__get__(self)
        self.tp_rank = 0
        for name in ('_load_model_weight_or_group_weight_scale', '_load_w13', '_load_w2'):
            setattr(self, name, getattr(FusedMoE, name).__get__(self))
        with patch('vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.get_tensor_model_parallel_world_size', return_value=1):
            method.create_weights(self, num_experts=24, hidden_size=128,
                                  intermediate_size_per_partition=1792,
                                  params_dtype=torch.bfloat16,
                                  weight_loader=FusedMoE.weight_loader.__get__(self),
                                  global_num_experts=192)

class Model(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = torch.nn.Module()
        self.layers[0].mlp.experts = Experts(owner)
    def get_expert_mapping(self):
        return FusedMoE.make_expert_params_mapping('gate_proj', 'down_proj', 'up_proj', num_experts=192)

source_path = os.environ.get('M2_SOURCE_PROFILE', '/opt/vllm/vllm/model_executor/models/xllm.py')
original = source_path == '/opt/vllm/vllm/model_executor/models/xllm.py'
if original:
    from vllm.model_executor.models.xllm import XllmModel
    loaded_module = {'XllmModel': XllmModel}
else:
    loaded_module = {'__file__':source_path, '__name__':'m2fp8source'}
    code = pathlib.Path(source_path).read_text()
    exec(compile(code, source_path, 'exec'), loaded_module)
for rank in [3,4,7]:
    model = Model(rank); pool = model.layers[0].mlp.experts
    for _, param in model.named_parameters():
        param.data.zero_()
    print('RANK', rank, 'FACTORY', type(pool.quant_method).__name__, [(n,tuple(p.shape)) for n,p in model.named_parameters()], flush=True)
    assert 'layers.0.mlp.experts.w2_weight_scale' in dict(model.named_parameters())
    for proj, marker in [('gate_proj', 7), ('up_proj', 11), ('down_proj', 4)]:
        for field in ['weight','weight_scale']:
            if field == 'weight':
                shape = tuple(pool.w2_weight[0].shape) if proj == 'down_proj' else (1792, 128)
                dtype = torch.float8_e4m3fn
            else:
                shape = tuple(pool.w2_weight_scale[0].shape) if proj == 'down_proj' else (pool.w13_weight_scale.shape[1] // 2, pool.w13_weight_scale.shape[2])
                dtype = torch.float32
            val = torch.full(shape, float(marker), dtype=dtype)
            key = f'layers.0.mlp.experts.100.{proj}.{field}'
            result = loaded_module['XllmModel'].load_weights(model, [(key,val)])
            local = pool.expert_map[100].item()
            print('LOAD',rank,proj,field,'result',result,'local',local,flush=True)
            if rank == 4:
                half = pool.w13_weight.shape[1] // 2
                slot = {'gate_proj': slice(0,half), 'up_proj':slice(half,2*half)}.get(proj)
                target = pool.w2_weight if proj == 'down_proj' else pool.w13_weight
                scale = pool.w2_weight_scale if proj == 'down_proj' else pool.w13_weight_scale
                target = scale if field == 'weight_scale' else target
                expected = val if field == 'weight' else val.float()
                if proj == 'down_proj':
                    assert torch.equal(target[local],expected)
                elif proj == 'gate_proj':
                    assert torch.equal(target[local,:target.shape[1]//2],expected)
                else:
                    assert torch.equal(target[local,target.shape[1]//2:],expected)
            else:
                assert not result
                assert all((x == 0).all() for x in [pool.w13_weight,pool.w2_weight,pool.w13_weight_scale,pool.w2_weight_scale])
    print('CASE_PASS',rank,flush=True)
print('REAL_COMPRESSED_TENSORS_FACTORY_AND_EXPERT_LOCALITY_PASS', flush=True)
