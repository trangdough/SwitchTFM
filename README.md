# Switch Transformer

Log book to keep track of my Switch Transformer learning process

## Learning Checkpoint

- [x] Switch Transformer Architecture
- [ ] Implement Switch Transformer (single GPU)
- [ ] Load balancer
- [ ] Parallelism
  - [ ] Model Parallelism
  - [ ] Data Parallelism
  - [ ] Expert Parallelism

## Notebook

### Switch Transformer Architecture

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
- The embedding space for each *token* is 2048-dimensional

#### 2. LayerNorm before Attention

\[
h_1 = \text{LayerNorm}(X_\text{in})
\]

\[
\Rightarrow{} h_1 \in \mathbb{R}^{B \times S \times D}
\]

#### 3. Multi-Head Self-Attention

- Q, K, V projections: `(B, S, D)`
- Multi-head reshaping: `(B, H, S, D, D_head)`
- Attention $\rightarrow$ `(B, S, D)`
- Output projection $\rightarrow$ `(B, S, D)`

Final attention output:
$$\text{attn\_out} \in \mathbb{R}^{B \times S \times D}$$

#### 4. Attention Residual (Add in `Add + Normalize`)

$$x_{\text{attn}} = x_{\text{in}} + \text{attn\_out}$$

$$\Rightarrow x_\text{attn} \in \mathbb{R}^{B \times S \times D}$$

#### 5. LayerNorm before SwitchFFN (Normalize in `Add + Normalize`)

$$h_2 = \text{LayerNorm}(x_\text{attn})$$

$$\Rightarrow{ }h_2 \in \mathbb{R}^{B \times S \times D}$$

This $h_2$ is **fed into the Router**.

#### 6. Router: Assign Expert to Each Token

Router is a single linear layer mapping hidden state $h_2$ to expert logits:

$$h_2 \in \mathbb{R}^{B \times S \times D} \qquad W_r \in \mathbb{R}^{D \times E}$$

Router logits:

$$\text{router\_logits} = h_2 W_r \in \mathbb{R}^{B \times S \times E}$$

Softmax over `E` experts:

$$p \in \mathbb{R}^{B \times S \times E}$$

Select expert(s):

1. `expert_idx`: Integer expert ID

- **Top-1:** $\in \mathbb{R}^{B \times S}$
- **Top-k:** $\in \mathbb{R}^{B \times S \times k}$

2. `gate_vals`: Scalar softmax probability

- **Top-1:** $\in \mathbb{R}^{B \times S}$
- **Top-k:** $\in \mathbb{R}^{B \times S \times k}$

#### 7. Construct Binary Routing Tensor R

One-hot:
$$
R[b,s,e] = 1 \quad \text{if expert $e$ is assigned for token $(b,s)$}
$$

$$
R \in \mathbb{R}^{B \times S \times E}
$$

#### 8. Compute Slot Indices (per Expert)

One method is using cumulative sum trick:

```python
positions = R.cumsum(dim=1) - 1     # (B, S, E)
slots = positions * R               # mask invalid values
slot_idx = slots.long()             # (B, S, E)
```

#### 9. Mask Tokens that Overflow Capacity

Capacity:
$$C = \text{capacity\_factor} \times \big \lceil \frac{BS}{E} \big \rceil$$

Validity mask:

```python
valid = (slot_idx < C)  # (B, S, E)
```

#### 10. Dispatch Tensor D

Binary dispatch tensor &rarr; to pack tokens into expert batches
$$R[b,s,e,c] = 1 \quad \text{if token $(b,s)$ is routed to expert $e$ at slot $c$}$$

$$R \in \mathbb{R}^{B \times S \times E \times C}$$

#### 11. Dispatch: Token &rarr; Per-Expert Batches

Einsum:

```python
X_expert = torch.einsum("bsec,bsd -> ecd", R, h2)
```

Mathematically:

$$X_\text{expert}[e,c,:] = \sum_{b,s} R[b,s,e,c] \cdot h_2[b,s,:]$$

$$X_\text{expert} \in \mathbb{R}^{E \times C \times D}$$

Where each $\text{expert}$ gets a batch size of $C$

#### 12. Expert FFNs (`E` independent networks)

FFN for expert $e \in E$:

- $X_\text{expert}[e,:,:] \in \mathbb{R}^{C \times D}$
- $W_{1e} \in \mathbb{R}^{D \times d_{ff}}$
- $W_{2e} \in \mathbb{R}^{d_{ff} \times D}$

$$H_e = \sigma(X_\text{expert}[e,:,:] W_{1e}) \quad \in \mathbb{R}^{C \times d_{ff}$$

$$Y_e = H_e W_{2e} \quad \in \mathbb{R}^{C \times D}$$

Stacked on $E$ experts:
$$y_{\text{expert}} \in \mathbb{R}^{E \times C \times D}$$

#### 13. Combine: Expert Outputs &rarr; Tokens

Recall the above dispatch tensor $D$:

```python
y_tokens = torch.einsum("bsec,ecd -> bsd", D, y_expert)
```

Mathematically:
$$y[b,s,:] = \sum_{e,c} D[b,s,e,c] \cdot y_{\text{expert}[e,c,:]}$$

$$y_{\text{tokens}} \in \mathbb{R}^{B \times S \times D}$$

#### 14. Apply router `gate_vals`

Recall:

- `gate_vals` $\in \mathbb{R}^{B \times S \times k}$
Gate scales output of chosen expert:

```python
y_tokens = y_tokens * gate_vals.unsqueeze(-1)
```

**Note to self:**

- `unsqueeze(-1)`: Add a new dimension to tensor; `-1` = position where new dimension should be added.

#### 15. FFN Residual Connection &rarr; Output

Recall:

- $x_{\text{attn}} \in \mathbb{R}^{B \times S \times D}$
- $y_{\text{tokens}} \in \mathbb{R}^{B \times S \times D}$

$$x_\text{out} = x_\text{attn} + y_\text{tokens}$$

$$x_{\text{out}} \in \mathbb{R}^{B \times S \times D}$$
