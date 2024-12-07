import torch
import  torch.nn as nn
import torch.nn.functional as F



class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.combine = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        batch, time, dimention = q.shape
        n_d = self.d_model // self.n_head
        q, k ,v = self.q(q), self.k(k), self.v(v)

        q = q.view(batch, time, self.n_head, n_d) 
        # 分成几个头self.n_head，每个头几个维度n_d，
        # 简单来说就是对最后一维进行切分
        # 然后 使用permute进行维度交换
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / (n_d ** 0.5) # 这里是计算q和k的点积

        mask = torch.tril(torch.ones(time, time), diagonal=1) # 生成一个下三角矩阵左下角是一
        score = score.masked_fill(mask == 0, float('-inf')) # 将下三角矩阵左下角是一的位置替换为负无穷，
        # 是因为softmax，e-无穷是0，所以这里是-inf

        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, self.d_model)

        out = self.combine(score)

        return out
    


BatchSize = 128
Time = 64
Dimention = 512
X = torch.randn(BatchSize, Time, Dimention)

d_model = 512 # 我要把他映射到qkv空间中，要多少维度
n_head = 8   # 他有几个头
attention = multi_head_attention(d_model, n_head)
out = attention(X, X, X)
print(out.shape)

# Embedding
class TakenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model)
        self.d_model = d_model


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super().__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.device = device
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad = False

    # 接着我们来生成位置
        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(dim=1) # 我们增加了一个维度
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 偶数位置
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 奇数位置

        def forward(self, x):
            batch, seq_len = x.shape
            return self.encoding[:seq_len, :]