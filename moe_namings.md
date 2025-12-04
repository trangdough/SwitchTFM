# MoE Dictionary

Purpose: List namings of basic components of MoE architecture in SGLang

## Main Layer

- `FusedMoE[Module]`
  - Represents the entire MoE layer
  - Orchestrates `gate`, `dispatch`, `expert FFN`

## Gating / Routing

- `TopK[Module]` : Calculates
  - Router logits
  - Select `top-k` experts for each token
- `router_logits` (or `score`): Raw output scores from gating network
- `topk_ids`: Indices of selected experts
- `topk_weights`: Gate values (softmax probabilities) for each selected expert

## Dispatcher

- `TokenDispatcher` (e.g., `StandardDispatcher`, `MaybeTboDeepEPDispatcher`)
  - Organizes tokens before they are sent to experts
- `DispatchOutput`:
  - Result of dispatch step
  - Permuted `hidden_states`
  - Sorting indices
- `gather_indx`: Indices used to **group tokens by experts** (dispatch)
- `scatter_indx`: Indices used to **restore tokens to their original order** (combine)

## Runner

- `MoeRunnerCore` (e.g., `TritonRunnerCore`, `MarlinRunnerCore`):
  - Execute actual matmuls (GEMMs) for **experts**
  - `MoeRunnerBackend`
  - `RunnerInput`: Data bundle passed to the runner (e.g., `hidden_states`, `routing_data`)
  - `RunnerOutput`: Result from the runner (e.g., computed `hidden_states`)

## Weights

Mixtral naming convention:

- `w1 (or gate_proj)`: Gate projection weight
- `w3 (or up_proj)`: Up projetion weight
- `w2 (or down_proj)`: Down projection weight
- `w13`: Fused `w1` and `w3` weights (often stored together for efficiency in `MergedColumnParallel`)

## Quantization Info

- `MoeQuantInfo`: A dataclass holding quantization-specific metadata (scales, zero-points, etc.) required by the runner.
