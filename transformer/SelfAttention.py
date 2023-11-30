import torch
import torch.nn as nn
import numpy as np
from MultiHeadAttention import MultiHeadAttention

class SelfAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super(SelfAttention, self).__init__()

        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))
        print("wq shape:", self.wq.shape)
        print("wk shape:", self.wk.shape)
        print("wv shape:", self.wv.shape)

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1./np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)
        print("q shape: ", q.shape)
        print("k shape: ", k.shape)
        print("v shape: ", v.shape)

        output, attn = self.mha(q, k, v, mask)
        return output, attn
    
if __name__ == "__main__":
    batch = 5
    n_head = 8
    n_x = 4
    d_k, d_v = 128, 64
    d_x , d_o = 80, 80

    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()

    selfattn = SelfAttention(n_head=n_head, d_k=d_k, d_v=d_v, d_x=d_x, d_o=d_o)

    output, attn = selfattn(x, mask)

    print(attn.size())
    print(output.size())
    