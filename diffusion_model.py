import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinosoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinosoidalPosEmb, self).__init__() # 调用父类的初始化方式
        self.dim = dim
        

    def forward(self, x):
        # x = x + self.pos_emb
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emd = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emd = x[:, None] * emd[None, :]
        emd = torch.cat([torch.sin(emd), torch.cos(emd)], dim=-1)
        return emb

# 我们自己写一个去噪神经网络，使用MLP
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        # 第一个神经网络对时间的编码
        self.time_mlp = nn.Sequential(
            # 对时间维度进行一个位置编码 
            SinosoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),# 在diffusion 中一般使用Mish激活函数
            nn.Linear(t_dim*2, t_dim)
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        # x: [batch_size, state_dim]
        # time: [batch_size, t_dim]
        # state: [batch_size, state_dim]
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        return x
    

class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule = 'linear', clip_denoised = True, **kwargs):
        super(Diffusion, self).__init__()

        self.state_dim = kwargs['obs_dim']
        self.action_dim = kwargs['act_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.device = torch.devise(kwargs['device'])
        self.T = kwargs['T']

        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02,self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)  # [1, 2, 3]  -> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)


        # 想一下，需要把参数都注册到模型中
        # 这样在训练的时候，就可以通过model.parameters()来获取所有的参数
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 前向过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 反向过程
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        





if __name__ == '__main__':
    pass