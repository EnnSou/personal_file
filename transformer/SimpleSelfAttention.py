import torch
import torch.nn as nn
import numpy as np
from ScaleDotProdutAttention import ScaledDotProductAttention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.d_q = d_model
        self.d_k = d_model
        self.d_v = d_model

        self.fc_q = nn.Linear(d_model, n_head * d_model)
        self.fc_k = nn.Linear(d_model, n_head * d_model)
        self.fc_v = nn.Linear(d_model, n_head * d_model)
        self.fc_o = nn.Linear(n_head * d_model, d_model)

        self.attention = ScaledDotProductAttention(scale=d_model)

    def forward(self, x, mask=None):
        batch, n_x, d_x = x.size()
        n_q = n_k = n_v = n_x

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        q = q.view(batch, n_q, self.n_head, self.d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_q)
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
    n_x = 4
    d_x = 80

    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()

    model = MultiHeadSelfAttention(n_head=8, d_model=d_x)
    output, attn = model(x, mask)

    print("attention size: ",attn.size())
    print("otput size: ", output.size())
