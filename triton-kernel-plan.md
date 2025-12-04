## Triton-Kernels Runner Refactor Plan


### Goal

Unify the Triton-kernels MoE backend with the new `MoeRunner` architecture by migrating the logic from `triton_kernels_moe.py` and quantization call sites into `triton_kernels.py`, delivering parity with the existing Triton runner while keeping routing, quantization, and dispatcher flows consistent with the roadmap vision.

### Milestones & Tasks

#### 1. Runner Core Implementation
- Extend `triton_kernels.py`:
  - Import helpers from `triton_kernels_moe.py` (`triton_kernel_fused_experts`, `triton_kernel_fused_experts_with_bias`, `triton_kernel_moe_forward`, `triton_kernel_moe_with_bias_forward`).
  - Flesh out `TritonKernelsRunnerInput`, `TritonKernelsRunnerOutput`, and `TritonKernelsQuantInfo` to carry routing data, biases, scale tensors, and optional zero-points.
  - Implement `TritonKernelsRunnerCore.run()`:
    - Branch on presence of biases or precision configs to call either `triton_kernel_fused_experts` (for bias-free execution) or `triton_kernel_fused_experts_with_bias` (when bias tensors or `PrecisionConfig` objects are provided).
    - Pass through `activation` and `apply_router_weight_on_input` directly to these helpers, and ensure `no_combine` plus `routed_scaling_factor` are applied in the post-run combine stage (mirroring the existing logic in `fused_moe_triton/layer.py`).
    - Flag mapping checklist:
      - `triton_kernel_fused_experts` accepts `activation` and `apply_router_weight_on_input`.
      - `triton_kernel_fused_experts_with_bias` accepts `activation`, `apply_router_weight_on_input`, and bias-specific precision configs (`w1_pcg`, `w2_pcg`).
      - `fused_experts` in `fused_moe_triton/fused_moe.py` consumes `no_combine`, `routed_scaling_factor`, `gemm1_alpha`, and `gemm1_clamp_limit` during the combine stage.
- Define fused and permute decorators:
  - Fused path for `DispatchOutputFormat.STANDARD` + `MoeRunnerBackend.TRITON_KERNEL` using `fused_experts` from `triton_kernels_moe.py` to avoid redundant permute logic when inputs already match the expected format.
   - Concrete refactored implementation:
      - Register `@register_fused_func("none", "triton_kernel")` in `triton_kernels.py`, mirroring the existing `triton.py` registration.
       - Inside the fused handler (`fused_experts_none_to_triton_kernels`), call `triton_kernel_moe_forward` (or `_with_bias`) with `dispatch_output.hidden_states`, `dispatch_output.topk_output`, and `quant_info` so the helper invokes `triton_kernel_fused_experts` directly.
      - Wrap the returned tensor in `StandardCombineInput` and bypass the pre/post permute stages, preserving the current zero-copy fast path used by the quantization methods.
      - Return `None` from the fused handler when `runner_config.no_combine` is set so `MoeRunner` falls back to the generic pre→run→post pipeline (the Triton-kernel helper only supports combined outputs today).
   - Concrete current implementation: 
      - `UnquantizedFusedMoEMethod.forward_cuda` branches on `self.use_triton_kernels` and calls `self.triton_kernel_moe_forward(...)` directly on the `StandardDispatchOutput` tuple. (`python/sglang/srt/layers/quantization/unquant.py`, lines 263-286). Then `triton_kernel_moe_forward` destructures the `StandardTopKOutput` tuple into `routing_data`, `gather_idx`, and `scatter_idx`, and passes those values directly to `triton_kernel_fused_experts` without additional permutation. (`python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, lines 54-98)
      - The MXFP4 quant path mirrors this logic: `Mxfp4MoEMethod.forward_cuda` calls `self.triton_kernel_moe_forward(...)` or its bias variant whenever `self.use_triton_kernels` is true. (`python/sglang/srt/layers/quantization/mxfp4.py`, lines 688-729)
  - Pre-permute taking `TritonKernelTopKOutput` and constructing a `TritonKernelsRunnerInput` dataclass that holds `dispatch_output.hidden_states`, `routing_data`, `gather_indx`, and `scatter_indx`.
    - This hook performs no additional permutations; it exists purely to satisfy the runner interface by packaging values into the dataclass.
    - Concrete refactored implementation: (This hook performs no permutations and exists purely to satisfy the runner interface)
      - Register `@register_pre_permute("triton_kernel", "triton_kernel")` in `triton_kernels.py`.
      - Build `TritonKernelsRunnerInput` by copying `dispatch_output.hidden_states` and decomposition of `dispatch_output.topk_output` (`routing_data`, `gather_indx`, `scatter_indx`).
      - Leave `running_state` untouched; the existing Triton-kernels helpers do not consume any cached state today. 
    - Concrete current implementation: no dedicated pre-permute exists today—quantization methods bypass it by passing `dispatch_output.topk_output` directly into `triton_kernel_moe_forward`, which internally accesses `routing_data`, `gather_indx`, and `scatter_indx`. (`python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, lines 54-98)
  - Post-permute returning `StandardCombineInput(hidden_states=...)` so dispatcher combine logic remains identical to other backends.
    - Concrete refactored implementation: (This hook performs no permutations and exists purely to satisfy the runner interface)
      - Register `@register_post_permute("triton_kernel", "standard")` in `triton_kernels.py`.
      - Return `StandardCombineInput(hidden_states=runner_output.hidden_states)`
      - Concrete current implementation: no dedicated post-permute exists today—quantization methods manually construct `StandardCombineInput(hidden_states=output)` immediately after calling the Triton-kernels helpers. (`python/sglang/srt/layers/quantization/unquant.py`, lines 263-286; `python/sglang/srt/layers/quantization/mxfp4.py`, lines 705-729)
- Update routing metadata plumbing:
- Extend `DispatchOutputFormat`/`DispatchOutputChecker` in `token_dispatcher/base.py` with a `triton_kernel` variant so permute registries can resolve the new runner format.
  - Rely on the existing `TritonKernelTopKOutput` in `topk.py`, which `TopK.forward` already emits when the Triton-kernel backend is active, and plumb that format through the runner registries.

#### 2. MoeRunner Integration & Configuration
- Update `MoeRunner.__init__` to construct `TritonKernelsRunnerCore` when `runner_backend.is_triton_kernel()`.
  - Add a new `elif runner_backend.is_triton_kernel(): self.runner_core = TritonKernelsRunnerCore(config)` branch alongside the existing Triton and DeepGEMM clauses.
- Register `fused_experts_none_to_triton_kernels` with `@register_fused_func("none", "triton_kernel")` so `FusedOpPool` resolves the standard fused path for the new backend; if the lookup returns `None`, `MoeRunner.run()` automatically falls back to the pre/post permute pipeline.

#### 3. Quantization Method Refactor
- For each quantization method referencing `triton_kernels_moe` directly (`python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/quantization/mxfp4.py`, `test/srt/test_triton_fused_moe.py`):
  - Replace direct calls with `self.runner.run(...)`, constructing `TritonKernelsQuantInfo` using the fields required at each call site:
    - `unquant.py` forward path: `w13_weight`, `w2_weight`, `w13_bias` (optional), `w2_bias` (optional).
    - `mxfp4.py` forward path: pre-converted tensors (`w13_weight` or `w13_precision_config`, `w2_weight` or `w2_precision_config`) and biases if present.
    - Tests (`test/srt/test_triton_fused_moe.py`) should construct `TritonKernelsQuantInfo` with the same fields used in `unquant.py` (weights and optional biases), since they currently mirror that call pattern.
  - Migrate tensor reshaping and padding for Triton-kernels (for example, expert weight transposition in `FusedMoE._load_w13` when `self.use_triton_kernels` is true) into shared helper utilities or the pre-permute stage so quant methods no longer branch on backend specifics.
- Ensure quantization methods honor the global MoE runner selection: when `get_moe_runner_backend()` returns `AUTO`, default to `TRITON_KERNEL` only if the layer was configured for Triton kernels, otherwise keep `TRITON`.
- Remove redundant attributes (`self.triton_kernel_moe_forward`, `self.triton_kernel_moe_with_bias_forward`, `self.use_triton_kernels`) once the runner handles invocation.
- Lazily import `MoeRunner` / `TritonKernelsQuantInfo` inside quantization methods to avoid circular dependencies with the new runner module.

#### 4. Legacy Helper Consolidation
  - Evaluate `triton_kernels_moe.py` after the refactor:
    - Retain kernel-level helpers (`triton_kernel_fused_experts`, `triton_kernel_fused_experts_with_bias`, `quantize`, `downcast_to_mxfp`) in `triton_kernels_moe.py`.
  - Remove dead exports that are provably unused after the migration, including the following direct imports of Triton-kernel helpers:
    - `python/sglang/srt/layers/quantization/unquant.py`: `triton_kernel_moe_forward`, `triton_kernel_moe_with_bias_forward`.
    - `python/sglang/srt/layers/quantization/mxfp4.py`: `triton_kernel_moe_forward`, `triton_kernel_moe_with_bias_forward`.
    - `test/srt/test_triton_fused_moe.py`: `triton_kernel_moe_forward`.
- Update tests (`test/srt/test_triton_fused_moe.py`) to exercise the new runner path via `MoeRunner` instead of calling helpers directly.

#### 5. Verification Strategy
- Unit/Integration tests:
  - Expand existing MoE tests to run with `--moe-runner-backend triton_kernel`, covering both bias and non-bias cases.
    - Modify `test/srt/test_triton_fused_moe.py` to:
      - Instantiate a `FusedMoE` layer with `MoeRunnerBackend.TRITON_KERNEL` (both with and without biases), feed synthetic inputs through `FusedMoE.forward`, and capture the runner output.
      - Compute the reference result using the existing `torch_naive_moe` helper already defined in the test and assert equality with the runner output for the same inputs.
      - Add subtests that verify routed-scaling-factor application and `no_combine` output shapes so the new runner hooks are exercised.
      - Explicitly set the top-k operator to emit `TritonKernelTopKOutput` and cover both fused (`no_combine=False`) and fallback (`no_combine=True`) paths, asserting shapes in each case. Currently only asserting numeric equality in the fused case.
  - Add regressions for router-weight application (top-k=1 vs top-k>1) and quantized code paths (e.g., `mxfp4`).
    - Extend `test_triton_fused_moe.py` to cover both router-weight configurations:
      - Scenario 1: `top_k = 1`, `apply_router_weight_on_input = False`.
      - Scenario 2: `top_k = 2`, `apply_router_weight_on_input = True`, verifying the runner’s output matches the reference once the weighting is toggled.
    - Extend `test/srt/quant/test_w4a8_deepseek_v3.py` (or introduce `test/srt/quant/test_mxfp4_triton_kernel.py`) to:
      - Configure the server with `--quantization mxfp4 --moe-runner-backend triton_kernel` (and any required MXFP4 flags).
      - Launch a minimal inference (e.g., single prompt) and capture the logits/output tensor.
      - Launch the same inference with the previous backend (e.g., Triton or Cutlass) and assert the outputs are numerically close, ensuring MXFP4 continues to match reference behavior under the Triton-kernel runner.
- Benchmarks: Don't rerun `benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py` as calls the Triton kernel directly (no `MoeRunner` involvement), so performance should remain unchanged.

#### 6. No-Combine=True Enablement
- Runner plumbing for `no_combine=True`:
  - Update `fused_experts_none_to_triton_kernels` so that when `runner_config.no_combine` is set it calls `triton_kernel_moe_forward` with `scatter_indx=None`, reshapes the `[tokens * top_k, hidden]` tensor into `[tokens, top_k, hidden]`, and returns it through the existing `TritonKernelsRunnerOutput(hidden_states=...)`. 
    - Flow: take the flattened activations from `TritonKernelsRunnerOutput.hidden_states`, reshape with `view(batch_tokens, top_k, hidden_size)`, and reuse the same tensor in the combine step (no data copy).
    - Ensure the reshape happens before `post_permute_triton_kernels_to_standard` wraps the tensor so downstream code receives the already structured `[tokens, top_k, hidden]` payload.
  - Adjust `pre_permute_triton_kernel_to_triton_kernels` and `post_permute_triton_kernels_to_standard` to honor the per-expert dimension when `no_combine=True`, using `view`-based reshapes to keep zero-copy semantics. 
- Dispatcher integration:
  - Reuse the existing `StandardCombineInput` (or the tensor fallback already accepted by `StandardDispatcher.combine`) to hand back a `[tokens, top_k, hidden]` view







