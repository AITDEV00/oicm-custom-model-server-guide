#!/usr/bin/env python3
"""Delegate to the measured stock-vendor real-FP8 factory regression.

The historical r1-only dummy/tensor-loader test was misleading and removed.
Its local geometry was not genuine FP8, and expected remote-skip was wrong.
The delegated script imports the unmodified vendor XllmModel, real create_weights,
global expert map, values, and block scales for ranks3/4/7.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("m2-fp8-stock-vendor-regression.py"),
                   run_name="__main__")

# RETIRED_HISTORICAL_TEST_BELOW
"""
from pathlib import Path
import sys
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.xllm import XllmModel


class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_map = {0: torch.tensor(0), 1: torch.tensor(-1)}
        self.tp_size = 1
        self.quant_config = None
        self.moe_config = type("Cfg", (), {"is_act_and_mul": True})()
        self.quant_method = type("QM", (), {})()
        self._map_global_expert_id_to_local_expert_id = (
            FusedMoE._map_global_expert_id_to_local_expert_id.__get__(self)
        )
        self.tp_rank = 0
        for method in ("_load_model_weight_or_group_weight_scale", "_load_w13", "_load_w2"):
            setattr(self, method, getattr(FusedMoE, method).__get__(self))
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            param = torch.nn.Parameter(torch.zeros(1, 4, 4))
            param.weight_loader = FusedMoE.weight_loader.__get__(self)
            if name.endswith("scale"):
                # Vendor enum FusedMoeWeightScaleSupported.BLOCK.value
                param.quant_method = "block"
            setattr(self, name, param)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = torch.nn.Module()
        self.layers[0].mlp.experts = Experts()

    def get_expert_mapping(self):
        return FusedMoE.make_expert_params_mapping(
            "gate_proj", "down_proj", "up_proj", num_experts=2)


def main():
    model = Model()
    assert [n for n, _ in model.named_parameters()] == [
        "layers.0.mlp.experts.w13_weight", "layers.0.mlp.experts.w2_weight",
        "layers.0.mlp.experts.w13_weight_scale", "layers.0.mlp.experts.w2_weight_scale",
    ]
    pool = model.layers[0].mlp.experts
    for proj, number, w2 in (("down_proj", 4, True), ("gate_proj", 7, False),
                              ("up_proj", 11, False)):
        for suffix in ("weight", "weight_scale"):
            marker = float(number + (100 if suffix.endswith("scale") else 0))
            checkpoint_key = f"layers.0.mlp.experts.0.{proj}.{suffix}"
            XllmModel.load_weights(model, [(checkpoint_key,
                                           torch.full((4, 4), marker))])
    assert (pool.w2_weight == 4).all()
    assert (pool.w2_weight_scale == 104).all()
    assert (pool.w13_weight[:, :2] == 7).all()
    assert (pool.w13_weight[:, 2:] == 11).all()
    assert (pool.w13_weight_scale[:, :2] == 107).all()
    assert (pool.w13_weight_scale[:, 2:] == 111).all()

    # The real fused callback must retain global/local expert locality, so
    # an off-rank expert returns False here rather than overwriting expert 0.
    off_rank = XllmModel.load_weights
    try:
        off_rank(model, [("layers.0.mlp.experts.1.down_proj.weight_scale",
                          torch.full((4,4), 999.0))])
    except RuntimeError as exc:
        assert "rejected sparse expert tensor" in str(exc)
    else:
        raise AssertionError("Remote expert load result not checked explicitly")
    # Above actual expected non-local behavior must be separately recognized
    # from the strict value route tests; there should be no overwrite.
    assert (pool.w2_weight_scale == 104).all()
    try:
        XllmModel.load_weights(model, [("layers.0.mlp.experts.0.gate_proj.example",
                                        torch.zeros(4,4))])
    except ValueError as exc:
        assert "Unsupported sparse Xllm expert parameter" in str(exc)
    else:
        raise AssertionError("Unsupported sparse parameter silently ignored")
    print("VENDOR_SPARSE_TO_FUSED_VALUE_PLACEMENT_PASS")


"""
