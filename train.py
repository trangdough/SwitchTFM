import torch
import torch.nn as nn
from torch.nn import functional as F

B = 1
S = 2
D = 4
E = 3
C = 2
d_ff = 6
k = 2
num_heads = 2
d_head = int(D / num_heads)

X_input = torch.rand(B, S, D) # (B=1, S=2, D=1)

# WQ1 (D=4, d_head=2)
WQ1 = torch.rand(D, d_head)
# WK1 (D=4, d_head=2)
WK1 = torch.rand(D, d_head)
# WV1 (D=4, d_head=2)
WV1 = torch.rand(D, d_head)

# WQ2 (D=4, d_head=2)
WQ2 = torch.rand(D, d_head)
# WK2 (D=4, d_head=2)
WK2 = torch.rand(D, d_head)
# WV2 (D=4, d_head=2)
WV2 = torch.rand(D, d_head)

WO = torch.rand(num_heads * d_head, D)

# Q1 (B=2, S=3, d_head=2)
Q1 = torch.matmul(X_input, WQ1)
# K1 (B=2, S=3, d_head=2)
K1 = torch.matmul(X_input, WK1)
# V (B=2, S=3, d_head=2)
V1 = torch.matmul(X_input, WV1)

# Q1 (B=2, S=3, d_head=2)
Q2 = torch.matmul(X_input, WQ2)
# K1 (B=2, S=3, d_head=2)
K2 = torch.matmul(X_input, WK2)
# V1 (B=2, S=3, d_head=2)
V2 = torch.matmul(X_input, WV2)

# (B, S, d_head) -> (B, d_head, S)
K1T = torch.transpose(K1, dim0=-2, dim1=-1)
# (B, S, d_head) @ (B, d_head, S) = (B, S, S)
QK1 = torch.matmul(Q1, K1T)
QK1_norm = QK1 / torch.sqrt(torch.tensor(K1.size(-1)))
# dim=-1 because we want to know for every query (a row),
# how much attention it pays to every key (column).
# As such, probabilities must sum to 1 across the columns.
QK1_softmax = torch.softmax(QK1_norm, dim=-1)
# (B, S, S) @ (B, S, d_head) = (B, S, d_head)
attn1 = torch.matmul(QK1_softmax, V1)

# (B, S, d_head) -> (B, d_head, S)
K2T = torch.transpose(K2, dim0=-2, dim1=-1)
# (B, S, d_head) @ (B, d_head, S) = (B, S, S)
QK2 = torch.matmul(Q2, K2T)
QK2_norm = QK2 / torch.sqrt(torch.tensor(K2.size(-1)))
# dim=-1 because we want to know for every query (a row),
# how much attention it pays to every key (column).
# As such, probabilities must sum to 1 across the columns.
QK2_softmax = torch.softmax(QK2_norm, dim=-1)
# (B, S, S) @ (B, S, d_head) = (B, S, d_head)
attn2 = torch.matmul(QK2_softmax, V2)

# (B, S, D / num_heads) stacked on top (B, S, D / num_heads) = (B, S, d_head)
attn_concat = torch.concat([attn1, attn2], dim=-1)

# Why W_O? Every single feature in final attn_out is a weighted sum of all features from all heads.
attn_out = torch.matmul(attn_concat, WO)

# (B, S, D)
x_attn = X_input + attn_out

# Flatten x_attn (B * S, D)
x_attn_flat = x_attn.view(-1, D)

class Router(nn.Module):
    def __init__(self, input_dim, output_dim, k, softmax_dim=-1):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=softmax_dim)
    
    def forward(self, x):
        # router_logits (B * S, D) @ (D, E) = (B * S, E)
        router_logits = self.linear(x)
        # p (B * S, E); along dim=expert
        p = self.softmax(router_logits)

        topk = torch.topk(p, k=self.k, dim=-1)
        # expert_idx (B * S, k)
        expert_idx = topk.indices
        # gate_vals (B * S, k)
        gate_vals = topk.values

        return expert_idx, gate_vals
    
router = Router(input_dim=D, output_dim=E, k=2, softmax_dim=-1)
# expert_idx (B * S, k); gate_vals (B * S, k)
expert_idx, gate_vals = router(x_attn_flat)

x_attn_expanded = torch.repeat_interleave(x_attn_flat, repeats=k, dim=0)
# Flatten expert index along all tokens
expert_idx_flat = expert_idx.view(-1)
# Group expert indices together
_, expert_idx_sort = torch.sort(expert_idx_flat, stable=True)
# Permute x_attn so tokens go to their assigned expert
x_attn_expGrouped = x_attn_expanded[expert_idx_sort]
# Count # tokens / expert
counts = torch.bincount(expert_idx_flat, minlength=E)
# Split expanded tokens to their dedicated experts
X_expert = torch.split(x_attn_expGrouped, split_size_or_sections=counts.tolist(), dim=0)

class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
experts = nn.ModuleList([
            ExpertFFN(input_dim=D, hidden_dim=d_ff, output_dim=D)
            for _ in range(E)
        ])

expertFFN_out = []
for i, x_e in enumerate(X_expert):
    out = experts[i](x_e)
    expertFFN_out.append(out)
y_expert = torch.cat(expertFFN_out, dim=0)

# unsort_indices (B * S * k,)
unsort_indices = torch.argsort(expert_idx_sort)
# y_expert_restored (B * S * k, D)
y_expert_restored = y_expert[unsort_indices]

# y_expert_reshaped (B * S, k, D)
y_expert_reshaped = torch.reshape(y_expert_restored, (B * S, k, D))
# gate_vals_reshaped (B * S, k, 1)
gate_vals_reshaped = torch.reshape(gate_vals, (B * S, k, 1))

# weighted_expert_out (B * S, k, D)
weighted_expert_out = y_expert_reshaped * gate_vals_reshaped
y_tokens = torch.sum(weighted_expert_out, dim=1)
y_tokens = torch.reshape(y_tokens, (B, S, D))

# final_output (B, S, D)
final_output = x_attn + y_tokens