# Switch Transformer
Log book to keep track of my Switch Transformer learning process

## Learning Checkpoint:
- [x] Switch Transformer Architecture
- [ ] Implement Switch Transformer (single GPU)
- [ ] Load balancer
- [ ] Parallelism
    - [ ] Model Parallelism
    - [ ] Data Parallelism
    - [ ] Expert Parallelism

## Notebook
### Switch Transformer Block: End-to-End
- `B` = batch_size
- `S` = seq_len
- `D` = d_model
- `E` = num_experts
- `C` = capacity (per expert)
- `d_ff` = FFN hidden size

Typical values:
- `D = 2048`
- `d_ff = 8192`
- `E = 64`

#### 1. Input
The block receives $$X_{\text{input}} \in \mathbb{R}^{B \times S \times D}$$
Example: `(B=8, S=1024, D=2048)` means the block receives:
- A batch of 8 input *sequences*
- Each sequence has 1024 *tokens*
- The embedding space for each *token* is 2048-dimensional.