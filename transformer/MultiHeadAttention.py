import torch
import torch.nn as nn
import numpy as np
from ScaleDotProdutAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(d_k)
        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.view(batch, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = k.view(batch, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = v.view(batch, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        
        output, attn = self.attention(q, k, v, mask)

        output = output.view(self.n_head, batch, n_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)

        return output, attn
    
if __name__ == "__main__":
    batch = 5
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    d_q, d_k, d_v = 256, 256, 128
    n_head = 8

    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)
    mask = torch.zeros(batch, n_q, n_k).bool()
    
    model = MultiHeadAttention(n_head, d_k_, d_v_, d_k, d_v, d_o=128)
    output, attn = model(q, k, v, mask)

    print(attn.size())
    print(output.size())


