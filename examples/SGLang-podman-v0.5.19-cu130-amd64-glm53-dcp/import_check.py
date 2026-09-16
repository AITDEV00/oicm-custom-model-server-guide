import sglang
import sglang.srt.layers.dcp
import sglang.srt.layers.attention.dsa_backend
import sglang.srt.managers.scheduler
import sglang.srt.managers.scheduler_components.load_inquirer
import sglang.srt.mem_cache.kv_cache_configurator
import sglang.srt.mem_cache.memory_pool
import sglang.kernels.ops.attention.dcp_kernels
import sglang.kernels.ops.attention.dsa.transform_index
import sglang.kernels.ops.attention.fixup_zero_kv
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla
import sglang.srt.models.deepseek_v2
print("SGLang:", sglang.__version__)
print("GLM-5.x DSA/DCP backport imports: OK")
