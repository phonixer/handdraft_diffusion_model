import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weight = 1.0):

        loss = self._loss(pred, target)
        WeightedLoss = (loss * weight).mean()
        return WeightedLoss
    
class WeightedL1(WeightedMSELoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)
    
class WeightedL2(WeightedMSELoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__() # 调用父类的初始化方式
        self.dim = dim
        

    def forward(self, x):
        # x = x + self.pos_emb
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

# 我们自己写一个去噪神经网络，使用MLP
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim = 16):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        # 第一个神经网络对时间的编码
        self.time_mlp = nn.Sequential(
            # 对时间维度进行一个位置编码 
            SinusoidalPosEmb(t_dim),
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
        # print('forward', x.shape, time.shape, state.shape)
        # 输出类型
        # print(type(x), type(time), type(state))
        
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        return x
    

class Diffusion(nn.Module):
    def __init__(self, 
                 loss_type, 
                 beta_schedule = 'linear', 
                 clip_denoised = True,
                 predict_epsilon=True, 
                 **kwargs
                 ):
        super(Diffusion, self).__init__()

        self.state_dim = kwargs['obs_dim']
        self.action_dim = kwargs['act_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.device = torch.device(kwargs['device'])
        self.T = kwargs['T']
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(kwargs["device"])

        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02,self.T, dtype=torch.float32, device=self.device)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)  # [1, 2, 3]  -> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)


        # 想一下，需要把参数都注册到模型中
        # 这样在训练的时候，就可以通过model.parameters()来获取所有的参数

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 前向过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # 反向过程
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        # 在指导xT的情况下，如何一步求出x0的结果
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        # 在指导xT的情况下，如何一步求出x0的结果

        # 求均值的两个系数
        self.register_buffer('posterior_mean_coef1', 
    betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2', 
    (1.0 - alphas_cumprod_prev) * torch.sqrt(betas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        # print(extract(self.posterior_mean_coef1, t, x.shape))
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(
            self.posterior_log_variance_clipped, t, x.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance


    def predict_start_from_noise(self, x, t, pred_noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise
        )

    
    def p_mean_variance(self, x, t, state):

        # print(x.shape, t.shape, state.shape)


        pred_noise = self.model(x, t, state) # 这个是预测的噪声
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        # 在有些代码中，为了稳定，会对pred_noise进行clip
        if self.clip_denoised:
            pred_noise = torch.clamp(pred_noise, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, state):
        # Sample 的过程 Algorithm 2
        # 我们首先计算model的均值和方差
        batchsize, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)

        # 生成mask，这个mask的作用就是在最后一步不需要noise
        nonzero_mask = (1 - (t == 0).float()).reshape(batchsize, *((1,) * (len(x.shape) - 1)))

        return model_mean + torch.exp(0.5 * model_log_variance) * noise * nonzero_mask

    def p_sample_loop(self, state, shape, *args, **kwargs):
        # state: [batch_size, state_dim]
        # shape: [batch_size, state_dim]
        # 这里的shape是一个形状，就是state的形状
        device = self.device
        batch_size = state.shape[0]
        # 接着我们生成最原始的噪声
        x = torch.randn(shape, device=device, requires_grad=False)
        # 这边DQL需要用到这个梯度，写成TRUE，这里是DDPM标准实现方法
        self.diffusion_steps = []  # 用于保存每一步的结果

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size, ),i , device=device, dtype=torch.long)
            x = self.p_sample(x,t,state)
            self.diffusion_steps.append(x.clone())  # 保存每一步的结果

        return x




    def sample(self, state, *args, **kwargs):

        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        # 在ddpm中，我们需要初始化一个噪声，
        # 那么这个噪声的形状是多大呢，就是这个state的形状，在这里使用shape来表示
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1.0, 1.0), self.diffusion_steps  # 限制在-1.0到1.0之间
    

    # --------------------------training----------------------------#
    # 就是预测噪声和标签噪声作比较，然后计算loss
    #

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        # 对应前向传播过程中的采样过程
        sample = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return sample


    def p_losses(self, x_start, state, t, weights = 1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # 是我们生成的噪声标签
        x_recon = self.model(x_noisy, t, state) # 是我们预测的噪声
        
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss




    def loss(self, x, state, weights = 1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)



    def forward(self, state, *args,**kwargs):
        return self.sample(state, *args, **kwargs)




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"  # cuda
    batchsize = 256
    act_dim = 5
    # x = torch.randn(256, 2).to(device)  # Batch, action_dim
    # state = torch.randn(256, 11).to(device)  # Batch, state_dim

    # 生成 x 张量，并转换为浮点类型
    x = torch.arange(1, batchsize + 1, dtype=torch.float32).unsqueeze(1).repeat(1, act_dim).to(device)
    # 生成 state 张量，每个 batch 的值都相同，并转换为浮点类型
    state = torch.arange(1, batchsize + 1, dtype=torch.float32).unsqueeze(1).repeat(1, 11).to(device)
    x = x / x.max()
    state = state / state.max() 


    model = Diffusion(loss_type='l2', obs_dim=11, act_dim=act_dim, hidden_dim=256, device=device, T=10)
    result = model(state)  # Sample result
    
    loss = model.loss(x, state)
    
    # print(f"action: {result};loss: {loss.item()}")
    import matplotlib.pyplot as plt
    import torch.optim as optim



    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    # 训练模型
    model.train()
    for i in range(10000):
        loss = model.loss(x, state)
        loss.backward()
        print(f"loss: {loss.item()}")
        optimizer.step()
        optimizer.zero_grad()



    # 训练结束后绘制扩散过程的图像
    state_test = state[100:101,:]
    x_test = x[100:101,:]
    print(state_test)
    print(state.shape)
    print(state_test.shape)
    print(x_test)
    print(x_test.shape)

    action, diffusion_steps = model.sample(state_test)

    # 算下loss
    loss = model.loss(x_test, state_test)
    print(f"action: {action};loss: {loss.item()}")
    # 输出真值
    print(x_test)
    print(len(diffusion_steps))


    # 绘制扩散过程的图像
    num_steps = len(diffusion_steps)
    steps_to_plot = [int(i * num_steps / 10) for i in range(10)] + [num_steps - 1]
    x_test = x_test.cpu().detach().numpy().flatten()
    plt.figure(figsize=(15, 5))
    for step_idx in steps_to_plot:
        step = diffusion_steps[step_idx].cpu().detach().numpy().flatten()
        print(step)
        plt.scatter([step_idx] * len(step), step, label=f'Step {step_idx}')

    plt.scatter([steps_to_plot[-1]] * len(x_test), x_test, label='Ground Truth')
    print(action)
    print(action.shape)

    plt.title('Diffusion Process')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('diffusion.png')
    plt.show()
