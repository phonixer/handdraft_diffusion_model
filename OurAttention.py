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

    def forward(self, q, k, v, mask=None):
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

        if mask is not None:
            # mask = torch.tril(torch.ones(time, time), diagonal=1) # 生成一个下三角矩阵左下角是一
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

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        out  = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out# 从代码上就是对最后一维度进行归一化
    

# FFN
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x) # 意义，减少激活神经元的一些连接
        x = self.linear2(x)

        return x
        
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, device):
        super().__init__()
        self.token_embedding = TakenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, maxlen, device)
        self.LayerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(x)
        x = token_embed + position_embed
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super().__init__()
        self.attention = multi_head_attention(d_model, n_head)
        self.LayerNorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.FFN = PositionwiseFeedforward(d_model, ffn_hidden, drop_prob)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, mask = None):
        _x = x 
        x = self.attention(x, x, x, mask)

        x = self.dropout1(x)
        x = self.LayerNorm1(x + _x)

        _x = x
        x = self.FFN(x)

        x = self.dropout2(x)
        x = self.LayerNorm2(x + _x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super().__init__()
        self.mask_attention = multi_head_attention(d_model, n_head)
        self.LayerNorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.attention = multi_head_attention(d_model, n_head)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.FFN = PositionwiseFeedforward(d_model, ffn_hidden, drop_prob)
        self.LayerNorm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.mask_attention(dec, dec, dec, t_mask) # 掩码   下三角掩码，不看到未来时刻的信息
        x = self.dropout1(x)
        x = self.LayerNorm1(x + _x)

        # 如果我们不做cross attention，enc的输入就是none，
        # 如果我们做cross attention

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)  # 这个掩码是为了不看到pad的信息

            x = self.dropout2(x)
            x = self.LayerNorm2(x + _x)

        _x = x
        x = self.FFN(x)
        x = self.dropout3(x)
        x = self.LayerNorm3(x + _x)

        return x
    

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, maxlen, device) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, maxlen, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
 

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, maxlen, device) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, maxlen, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, dec, enc, t_mask, s_mask):
        x = self.embedding(dec)
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        dec = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, vocab_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, maxlen, device) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, maxlen, device)
        self.decoder = Decoder(vocab_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, maxlen, device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.shape[1], k.shape[1]

        # 首先我们要思考一下生成的mask的形状是什么样的
        # (Batch, time, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(1) # 这里不太懂
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)


        # 按位取与，只有两个都是1的时候才是1

        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1]
        mask = (torch.tril(torch.ones(len_q, len_k)) == 0).to(q.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) & self.make_casual_mask(trg, trg)
        # 交叉mask
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_trg_mask)

        return out