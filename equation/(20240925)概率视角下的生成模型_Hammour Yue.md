# 概率视角下的生成模型

 **Author:** [Hammour Yue]

 **Link:** [https://zhuanlan.zhihu.com/p/611466195]

作为一只刚入坑两个月生成模型的小白，初入该领域时被各种眼花缭乱的概念和公式弄得头晕眼花，例如KL散度、变分推断、变分下限等等。因此希望能通过这篇文章，以概率的视角，讨论、梳理一下目前主流生成模型的动机和核心思想，只讨论原始模型，不讨论改进方案，也不会聚焦于代码的实现。会试着从公式推导中，引导出各种假设、建模，最终确定模型，因此本文视角和原论文略有不同。文章的最终目的是作为自己的学习笔记，可能有许多错误或者对各个模型有理解不到位的地方，希望大家能够批评指正。

## 1 定义 Definition  
生成模型的一般定义是：给定从真实分布 $p(x)$ 采样的观测数据 $x\sim p(x)$ ，训练（参数估计）得到一个由 $\theta$ 控制逼近真实分布的 $p_\theta(x)$，称 $p_\theta(x)$ 为生成模型。

$x$ 一般可以认为是来自同一分布的图片，每一张图片是一个多维的向量。

$p(x)$ 代表数据真实的分布，观测的数据就是从真实分布中采样出来的。真实分布一般没有办法得到，哪怕我们熟知的抛硬币的例子，看起来似乎很正确的正反面概率为50%，但是真实分布很可能不是如此简单的，因为整个过程还包括空气的阻力，每次抛硬币的力度、角度，甚至太阳风暴、地球磁场等等都会影响抛硬币正反的概率（分布），所以真实分布是非常复杂的。

$p_\theta(x)$ 代表的就是我们建立的生成模型，例如上述抛硬币的例子，我们可以不考虑那些奇怪的因素，直接简单的建模成 $\theta$ 控制的伯努利分布，利用参数估计的方法得到参数 $\theta$ ，也可以考虑力度、角度等等建立更复杂而更加精确的模型。

在得到生成模型 $p_\theta(x)$ 后，就能采样（生成）出各式各样的，包括观测集没有的数据了，这就是最终的目的。

## 2 难点 Difficulties  
关于图像的 $p_\theta(x)$ 的建模肯定不如抛硬币的简单。一个直接的想法是，我们可以暴力的利用复杂的神经网络去拟合 $p_\theta(x)$ ，但是神经网络只能拟合一些简单的分布，例如高斯分布等，显然一般的高斯分布达不到拟合真实分布的能力。所以对 $p_\theta(x)$ 合适的建模是研究中最主要关心的问题。

一般而言，我们没有办法直接从 $p_\theta(x)$ 中去采样得到 $x$ ，而容易的是从均匀分布，或者高斯分布（重参数化技巧）中采样的到数据，如何利用简单分布或者利用其它的方式采样也是建模过程中需要考虑的。

## 3 模型 Models  
接下来主要从隐变量模型和基于能量函数的模型两个方面来介绍。

### 3.1 隐变量模型 Latent Variable Models  
真实分布 $p(x)$ 不好直接分析或者逼近，那我们可以把它拆开 $p(x)=\int p(x\mid z)p(z)dz$ ， $z$ 服从分布 $p(z)$ 。其中 $z$ 就叫做隐变量， $p(z)$ 是可以任意指定的先验概率分布，形如这样的模型就叫做**隐变量模型**。由于隐变量只有 $z$ 一个变量，在这里我们也可以叫做单隐变量模型。

对于单隐变量模型，只需要使 $p_\theta(x\mid z)$ 来逼近 $p(x\mid z)$ ，理论上就能利用公式 $p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$ 得到生成模型。“逼近”的衡量标准通常采用KL散度，即最小化 $KL(p(x\mid z) \| p_\theta(x\mid z))$ ，这个式子也不容易处理。

换种方式，对联合概率密度 $p(x,z)$ 的逼近显然也是可以的，这是由于 $p(x,z)=p(x\mid z)p(z)$ 而 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ ， $p(z)$ 为先验分布，这样就相当于拟合了 $p(x\mid z)$ 。此时，

$$ \begin{align} KL(p(x,z)\|p_\theta(x,z)) &=\iint p(x, z) \log \frac{p(x, z)}{p_\theta(x, z)} d z d x \\ &=\int p(x)\left[\int p(z \mid x) \log \frac{p(x) p(z \mid x)}{p_\theta(x, z)} d z\right] dx\\ &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{p(x) p(z \mid x)}{p_\theta(x, z)} d z\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \left[\log p(x) + \log \frac{p(z \mid x)}{p_\theta(x, z)}\right] d z\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[\log p(x) \int p(z \mid x) d z\right]+\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{ p(z \mid x)}{p_\theta(x, z)} d z\right]\\ \end{align}\tag{3.1} $$


由于第一项中，

$$\begin{align} \mathbb{E}_{x \sim p(x)}\left[\log p(x) \int p(z \mid x) d z\right]&=\mathbb{E}_{x \sim p(x)}\left[\log p(x)\right]\\ &=C \end{align}\tag{3.2}$$  


其中 $C$ 是一个常数，所以，

$$\begin{align} \mathcal{L}&=K L(p(x, z) \| p_\theta(x, z))-C\\ &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{p(z \mid x)}{p_\theta(x, z)} d z\right] \end{align}\tag{3.3}$$  


即最小化 $\mathcal{L}$ 等价于最小化 $KL(p(x,z)\|p_\theta(x,z))$ ，也可以理解为极大化 $-\mathcal{L}$ ，因此 $-\mathcal{L}$ 也被叫做**变分下限(ELBO)**，接着可以对 $\mathcal{L}$ 再进行一些变化：

$$\begin{align} \mathcal{L} &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{p(z \mid x)}{p_\theta(x, z)} d z\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{p(z \mid x)}{p_\theta(x\mid z)p(z)} d z\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[-\int p(z \mid x) \log p_\theta(x\mid z) dz+\int p(z \mid x) \log \frac{p(z \mid x)}{p(z)} d z\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z\mid x)}\left[-\log p_\theta(x\mid z) \right]+KL(p(z\mid x) \|p(z))\right]\\   \end{align}\tag{3.4}$$  
变化到这，可以看到我们引入了真实后验分布 $p(z\mid x)$ ，表示的是观测到 $x$ ，与之对应的隐变量 $z$ 的真实概率分布。既然这样，那我们还需要神经网络来拟合 $p_\theta(z\mid x)$ ，一般我们叫做编码器Encoder；与之对应，似然函数 $p_\theta(x\mid z)$ 叫做解码器（生成器）Decoder（Generator）。

首先看待优化函数 $\mathcal{L}$ 的第一项， $p_\theta(x\mid z)$ 是似然函数，那么最小化 $-\log p_\theta(x\mid z)$ 也就是最大化似然函数。第二项是希望真实后验分布 $p(z\mid x)$ 与先验分布 $p(z)$ 尽量接近，这样就能使先验分布采样的 $z$ 与真实的 $x$ 对应起来，放入Decoder后，更好进行生成。

至此，隐变量模型的大体框架就清晰易见了，用神经网络拟合 $p_\theta(z\mid x)$ 和 $p_\theta(x\mid z)$ ，最小化 $\mathcal{L}$。到这里可能有人会问，为啥直接用KL散度就能直接当做优化目标函数呢？一般来说，我们使用极大似然估计来作为优化目标，一般形式是 $\mathcal{L}=\log \prod_x p_\theta(x)=\sum_x{\log p_\theta(x)}$ 。但KL散度和极大似然估计本质上是“等价”的，对（3.3）式进行变形： $$\begin{align}  -\mathcal{L} &=C-K L(p(x, z) \| p_\theta(x, z))\\  &=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \log \frac{p_\theta(x, z)}{p(z \mid x)} d z\right]\\ &= \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)  \log \frac{p_\theta(x)p_\theta(z\mid x)}{p(z \mid x)} dz  \right]\\ &=\mathbb{E}_{x \sim p(x)}[\log p_\theta(x)-KL(p(z \mid x)\|p_\theta(z\mid x))] \end{align}\tag{3.3*}$$  
可以看到第一项为对数似然，第二项为大于等于0的数，因此对数似然大于等于 $-\mathcal{L}$ 变分下限。而严格来说， $KL(p(z \mid x)\|p_\theta(z\mid x))$ 是很难取到0的，因为对变分下限的优化是尽量让 $p_\theta(z\mid x)$ 与 $p(z)$ 相近，而真实后验 $p(z\mid x)$ 与先验 $p(z)$ 的形式不一样，所以在后面的VAE或者扩散模型中，其损失函数没法训练一个“真正”的最大的似然，也有些文章对训练的似提供了改进，这篇文章不做讨论了。

那这样的话还有几个问题，第一个是之前**2**部分提到的，神经网络一般只能拟合简单的分布，例如高斯分布，那这里应该怎么做呢？第二个就是 $x$ 要如何生成呢，如果直接利用公式 $p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$ ，那么岂不是要采样许多（无穷多）的 $z$ ，并且得到了 $p_\theta(x)$ 也不知道如何去采样。

其实可以换种视角来理解， $p_\theta(x\mid z)$ 意味着给定一个 $z$ 输出与之对应 $x$ 的分布，此时采样得到的 $x$ 就可以认为是由 $z$ 生成的，但是采样的问题仍然需要解决。但如果这个分布的期望（均值）为 $\mu$ ，那么也完全可以用 $\mu$ 来表示由 $z$ 生成的 $x$ ，这样采样的问题也能解决了。

所以现在的要求是：

**1、** $p_\theta(x\mid z)$ 的形式要容易计算，要算出极大似然，并且均值要容易得到，方便生成；

**2、**选择合适后验分布 $p(z\mid x)$ 的形式，使得能与先验 $p(z)$ 的KL散度容易计算。

自此就是单隐变量模型的归一化表述了，具体的模型需要做不同的假设和分析。而对于多隐变量模型，会在3.1.3 Diffusion部分进行说明。

### 3.1.1 自编码器 AE  
自编码器的思想是隐变量 $z$ 和 $x$ 能唯一对应， $x$ 到 $z$ 属于特征降维， $z$ 到 $x$ 属于特征还原，相当于高维空间和低维流形之间的映射，属于单隐变量模型，且隐变量空间是连续均匀的。

既然是唯一一对应，那么真实后验分布 $p(z\mid x)= \delta(z-C(x))$，这里的 $\delta$ 指的是狄拉克函数， $C$ 代表真实编码器，即 $C(x)$ 和后验分布 $p(z\mid x)$ 的均值是相互对应的。幸运的是这个分布用神经网络也是很好拟合的，这是因为 $p_\theta(z\mid x)=\delta(z-C_\theta(x))$ ，神经网络的输入是 $x$ ，输出就能直接表示成均值 $C_\theta(x)$ 了。所以实际上拟合的是 $C(x)$ 这个函数，拟合函数当然是神经网络最在行的事情了，自然也就拟合了这个分布。而先验分布假设 $p(z)=\frac{1}{D}, 0<z<D$ ， $D$ 为某个正数。这样其实就满足了要求**2。**

关于似然函数的拟合，跟之前一样，也采用狄拉克函数 $p_\theta(x\mid z)=\delta(x-G_\theta(z))$ ，其中 $G_\theta$ 是生成器（解码器），这样要求**1**也就都能满足了。

为了计算（3.4），我们可以将似然函数 $p_\theta(x\mid z)=\delta(x-G_\theta(z))$ 看成是均值为 $G_\theta(x)$ ，方差无限接近0的高斯分布 $\lim_{{\sigma} \rightarrow 0}{N(G_\theta(x),{\sigma}^2)}$ 。类似的，后验分布 $\delta(z-C(x))$ 也可以看成是均值为 $C(x)$ ，方差无限接近0的高斯分布 $\lim_{\hat{\sigma} \rightarrow 0}{N(C(x),\hat{\sigma}^2)}$ ，这样损失函数 $\mathcal{L}$ 的第一项：

$$\begin{align} \mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z\mid x)}\left[ -\log p_\theta(x\mid z) \right]\right] &=\mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z\mid x)}\left[ -\log \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-G_\theta(z))^2}{2\sigma^2}}  \right]\right]\\ &=\mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z\mid x)}\left[ \frac{(x-G_\theta(z))^2}{2\sigma^2}  \right]\right] - \log \frac{1}{\sqrt{2\pi}\sigma} \\ &=\frac{1}{2\sigma^2}\mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z\mid x)}\left[ (x-G_\theta(z))^2  \right]\right]-\log \frac{1}{\sqrt{2\pi}\sigma}\\ \end{align}\tag{3.5}$$  
由于 $\sigma$ 是与 $\theta$ 无关的定值，所以 $\mathcal{L}$ 的第一项显然是熟悉的MSE损失函数。那么其第二项：

$$\begin{align} KL(p(z\mid x) \|p(z)) &=\int \delta(z-C(x))\log \frac{\delta(z-C(x))}{p(z)}dz \\ &=\int \delta(z-C(x))\log \delta(z-C(x))dz-\int\delta(z-C(x)) \log p(z)dz\\ &=\int \delta(z-C(x))\log \delta(z-C(x))dz-\log p(C(x))\\ &=\int \delta(z-C(x))\log \delta(z-C(x))dz +\log D \end{align}\tag{3.6}$$  


对于前一项有：

$$\begin{align} \int \delta(z-C(x))\log \delta(z-C(x))dz &=\int \frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}} \log \frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz \\ &=\int -\frac{(z-C(x))^2}{2\hat{\sigma}^2}\frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz + \log \frac{1}{\sqrt{2\pi}\hat{\sigma}}\int \frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz\\ &=-\frac{1}{2}-\log \sqrt{2\pi}\hat{\sigma} \end{align}\tag{3.7}$$  


所以 $KL(p(z\mid x) \|p(z))=-\frac{1}{2}-\log \sqrt{2\pi}\hat{\sigma}+\log D$ ，这显然也是与 $x,\theta$ 无关的一个常量，优化过程中可以不用考虑。

到此为止，自编码器AE的目标函数就推导出来了，即 $\mathcal{L}$ 为MSE。实际上 $x$ 应该是服从独立多维分布的（每个像素点），但是推导形式和一维的没有本质区别，最后多维的目标函数就是一维目标函数的和。

### 3.1.2 变分自编码器 VAE[1]  
VAE认为一个 $x$ 应该对应一个分布，这个分布又能重新生成 $x$ 。那所以生成的时候应该用一个分布生成一个 $x$ ？这显然不现实。但是在这个分布中采样一个点来生成 $x$ 是可行的，这是因为采样这个过程蕴含了该分布的统计学信息。具体而言， $x$ 对应的分布通过采样生成隐变量 $z$ ，最后通过真实解码器应该又能生成 $x$ 。

因此，为了生成的时候方便采样，先验分布设为标准高斯分布 $p(z)= N(0,1)$ ，而真实后验分布应该为 $\mu=C(x)$的高斯分布 $p(z\mid x)= N(\mu,\sigma^2)$ ，含义就是一个 $x$ 对应一个高斯分布。所以对于 $p_\theta(z\mid x)$ ，直接用神经网络拟合函数 $C_\theta(x)$ ，此时 $p_\theta(z\mid x)= N(C_\theta(x), \sigma^2)$ ，接下来 $p_\theta(x\mid z)$ 就不做变化了。那么损失函数 $\mathcal{L}$ 的第一项似然函数和AE的推导是完全一致的，为MSE。关键的部分是第二项的KL散度：

$$\begin{align} KL(p(z\mid x) \|p(z)) &=\int N(\mu,\sigma^2)\log \frac{N(\mu,\sigma^2)}{N(0,1)}dz \\ &=\int \frac{1}{\sqrt{2 \pi }\sigma} e^{-(z-\mu)^{2} / 2 \sigma^{2}}\left(\log \frac{e^{-(z-\mu)^{2} / 2 \sigma^{2}} / \sqrt{2 \pi}\sigma}{e^{-z^{2} / 2} / \sqrt{2 \pi}}\right)dz\\ &=\int \frac{1}{\sqrt{2 \pi }\sigma} e^{-(z-\mu)^{2} / 2 \sigma^{2}} \log \left\{\frac{1}{\sigma} \exp \left\{\frac{1}{2}\left[z^{2}-(z-\mu)^{2} / \sigma^{2}\right]\right\}\right\} dz\\ &=\frac{1}{2} \int \frac{1}{\sqrt{2 \pi }\sigma} e^{-(z-\mu)^{2} / 2 \sigma^{2}}\left[-\log \sigma^{2}+z^{2}-(z-\mu)^{2} / \sigma^{2}\right] dz\\ &=\frac{1}{2}\left(-\log \sigma^2+\mu^2+\sigma^2-1 \right) \end{align}\tag{3.8}$$  
到此，VAE的目标函数也推导出来了，与AE的区别仅仅是后验分布和先验分布的假设。

在此视角下，我们其实是用 $p_\theta(z\mid x)$ 拟合 $p(z\mid x)$ 。又因为隐变量模型的KL散度约束，又使得 $p(z\mid x)$ 逼近$p(z)$ 。所以对 $p(z\mid x)$ 的约束，实际也是对 $p_\theta(z\mid x)$ 的约束，也就是$p_\theta(z\mid x)$逼近 $p(z)$ 。

训练的时候，因为每个 $x$ 都对应一个高斯分布，看起来好像有很多不同的高斯分布都想逼近 $p(z)$。那么在采样时候，一个 $z$ 可能会被很多的高斯分布去抢，可能离谁的均值最近就像谁，最后可能会成为“四不像”怪物，这就达到了生成多样性的目的了。

### 3.1.3 扩散模型 Diffusion[2]  
扩散模型的目标和VAE是一样的，都是希望一个 $x$ 对应一个分布，这个分布和先验之间会有约束。生成的时候就能从先验分布中采样生成不同的 $x$ 了。也即真实后验分布为 $p(z_T\mid x)=N(\mu,\sigma^2)$ ，先验 $p(z_T)=N(0,1)$ 。和VAE的区别在于， $x$ 和 $z_T$ 之间又添加了许多的隐变量，因为这样能使模型更复杂，理论上效果会更好。如果将生成模型 $p(x)$ 拆成多隐变量积分： 

$$\begin{align} p(x) &=\int...\int p(x,z_1,z_2,z_3,...,z_T)dz_1dz_2...dz_T\\ &=\int p(x,z_{1:T})dz_{1:T}\\  &=\int p(x\mid z_{1:T})p(z_1\mid z_{2:T})...p(z_T)dz_{1:T}\\ \end{align}\tag{3.9}$$  
但是这样的模型不太好处理，所以扩散模型还用了马尔科夫假设，隐变量之间成链状关系，且每个隐变量只跟相邻的相关，所以最终：$$p(x)=\int p(x\mid z_1)p(z_1\mid z_2)p(z_2\mid z_3)...p(z_T)dz_{1:T}\tag{3.10}$$  


和单隐变量模型处理一样，我们仍然考虑联合分布的逼近：

$$\begin{align} KL(p(x,z_{1:T})\|p_\theta(x,z_{1:T})) &=\int p(x,z_{1:T})\log \frac{p(x,z_{1:T})}{p_\theta(x,z_{1:T})}dz_{1:T}dx\\ &=\int p(x)p(z_{1:T}\mid x)\log \frac{p(x)p(z_{1:T}\mid x)}{p_\theta(x,z_{1:T})}dz_{1:T}dx\\ &=\mathbb{E}_{x\sim p(x)} \left[\int p(z_{1:T}\mid x)\log \frac{p(x)p(z_{1:T}\mid x)}{p_\theta(x,z_{1:T})}dz_{1:T} \right]\\ &=\mathbb{E}_{x\sim p(x)} \left[\int p(z_{1:T}\mid x)\log \frac{p(z_{1:T}\mid x)}{p_\theta(x,z_{1:T})}dz_{1:T} \right]+C \end{align}\tag{3.11}$$  
这里和单隐变量模型的推导非常近似，那么显然前一项就是多隐变量模型的变分下限 $\mathcal{L}$ 。同样的，优化 $\mathcal{L}$ 就相当于优化KL散度，也相当于极大似然估计了。为了方便书写，令 $\mathcal{P}$ 为 $\mathcal{L}$ 期望里面的积分，接着再对 $\mathcal{P}$ 进行一些变化：

$$\begin{align} \mathcal{P} &=\int p(z_{1:T}\mid x)\log \frac{p(z_{1:T}\mid x)}{p_\theta(x,z_{1:T})}dz_{1:T}\\  &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)}\left[\log \frac{p(z_{1:T}\mid x)}{p_\theta(x,z_{1:T})} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)}\left[\log \frac{p(z_1\mid x)\prod_{t=2}^{T}p(z_t\mid z_{t-1})}{p(z_T)p_\theta(x\mid z_1)\prod_{t=2}^{T}p_\theta(z_{t-1}\mid z_{t})} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ -\log p(z_T)+\log \frac{p(z_1\mid x)}{p_\theta(x\mid z_1)} +\sum_{t=2}^{T}{\log \frac{p(z_t\mid z_{t-1})}{p_\theta(z_{t-1}\mid z_t)}} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ -\log p(z_T)+\log \frac{p(z_1\mid x)}{p_\theta(x\mid z_1)} +\sum_{t=2}^{T}{\log \frac{p(z_t\mid z_{t-1},x)}{p_\theta(z_{t-1}\mid z_t)}} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ -\log p(z_T)+\log \frac{p(z_1\mid x)}{p_\theta(x\mid z_1)} +\sum_{t=2}^{T}{\log \left[\frac{p(z_{t-1}\mid z_t,x)}{p_\theta(z_{t-1}\mid z_t)}\cdot \frac{p(z_t\mid x)}{p(z_{t-1}\mid x)} \right]} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ -\log p(z_T)+\log \frac{p(z_1\mid x)}{p_\theta(x\mid z_1)} +\sum_{t=2}^{T}{\log \frac{p(z_{t-1}\mid z_t,x)}{p_\theta(z_{t-1}\mid z_t)}+\sum_{t=2}^{T}{\log \frac{p(z_t\mid x)}{p(z_{t-1}\mid x)}}} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ -\log p(z_T)+\log \frac{p(z_1\mid x)}{p_\theta(x\mid z_1)} +\sum_{t=2}^{T}{\log \frac{p(z_{t-1}\mid z_t,x)}{p_\theta(z_{t-1}\mid z_t)}+\log \frac{p(z_T\mid x)}{p(z_1\mid x)}} \right]\\ &=\mathbb{E}_{z_{1:T}\sim p(z_{1:T}\mid x)} \left[ \log \frac{p(z_T\mid x)}{p(z_T)} - \log p_\theta(x\mid z_1)+\sum_{t=2}^{T}{\log \frac{p(z_{t-1}\mid z_t,x)}{p_\theta(z_{t-1}\mid z_t)}} \right]\\ &=KL(p(z_T\mid x)\|p(z_T))+\mathbb{E}_{z_1\sim p(z_1\mid x)}\left[-\log p_\theta(x\mid z_1) \right] + \sum_{t=2}^{T}\mathbb{E}_{p(z_t\mid x)}[{KL(p(z_{t-1}\mid z_t,x)\|p_\theta(z_{t-1}\mid z_t))}]\\ \end{align}\tag{3.12}$$  
 因此目标函数 $\mathcal{L}$ 就为：

$$\mathcal{L}=\mathbb{E}_{x\sim p(x)}\left[KL(p(z_T\mid x)\|p(z_T))+\mathbb{E}_{z_1\sim p(z_1\mid x)}\left[-\log p_\theta(x\mid z_1) \right] + \sum_{t=2}^{T}\mathbb{E}_{p(z_t\mid x)}[{KL(p(z_{t-1}\mid z_t,x)\|p_\theta(z_{t-1}\mid z_t))}]\right]\tag{3.13}$$  
可以看到，在马尔科夫假设下多隐变量模型，最终的目标函数和单隐变量模型的非常相似，都是KL散度+似然的形式。显然，目标函数里存在大量的似然函数 $p(z_{t-1}\mid z_t)$ ，如果每个分布都用一个神经网络来模拟，那样会非常复杂，开销也会非常大，因此就需要对模型做出一些假设。

而且如果所有隐变量之间的似然函数都采用神经网络来拟合的话，也是很复杂的，所以扩散模型就先直接假设后验分布（前向分布）为高斯分布 $p(z_t\mid z_{t-1})=N(\sqrt\alpha_t z_{t-1}, \beta_t)$ ，其中 $\beta_t=1-\alpha_t$ 。 $\alpha_t$ 是事先规定好的，一般会让其越来越小，趋向于0。它的均值和方差是固定的，没有需要学习的参数，所以不需要用神经网络来拟合，这样就简化了模型。为什么假设这样的前向分布，又使用这样奇怪的 $\alpha_t$ ？这是因为：

$$\begin{align} p(z_T\mid x) &=p(z_1\mid x)\prod_{t=2}^{T}p(z_t\mid z_{t-1})\\ &=N(z_1;\sqrt\alpha_0 x, \beta_0)\prod_{t=1}^{T}N(z_t;\sqrt\alpha_t z_{t-1}, \beta_t)\\ &=N(z_T;\sqrt{\bar{\alpha}_t}x,1-\bar{\alpha}_t) \end{align}\tag{3.14}$$  
其中 $\bar{\alpha}=\prod_{t=1}^{T}\alpha_t$ ，可以发现当 $T$ 足够大时，真实后验分布 $p(z_T\mid x)$ 会足够接近先验 $p(z_T)$ 标准高斯分布，这样在不用神经网络的情况下就完成了“一个 $x$ 对应一个分布并且这个分布不断接近先验分布”的过程，也就解决了目标函数 $\mathcal{L}$ 的第一项，没有学习的参数，且近似为0。这个思想跟VAE的也是一样的，使得后验分布不断逼近标准高斯分布。

现在看来损失函数后面两项，对于似然 $p_\theta(z_{t-1}\mid z_t),t\in[0,T]$ ，$x$ 可以看做 $z_0$ ，和VAE一样也采用高斯分布 $\mu=G_{\theta,t}(z_t)$， $p_\theta(z_{t-1}\mid z_t)=N(\mu,\sigma^2)$ ，方差和VAE一样固定住，不参与估计。但是看起来好像每个分布都需要用神经网络去拟合？其实不然，只要将 $t$ 作为神经网络的一个输入，即输入为 $(t,z_t)$ ，输出为 $(G_{\theta,t}(z_t),\sigma)$ ，就能同时拟合所有的似然了。那么损失函数第二项的后验就是 $G_{\theta,1}(z_1)$ 和 $x$ 之间的MSE。

第三项显然是约束我们拟合隐变量的似然，要求与真实的似然相近，这也是非常合理的要求。主要的问题是需要将 $p(z_{t-1}\mid z_t,x)$ 的具体形式表示出来，对其进行变化：

$$\begin{align} p(z_{t-1}\mid z_t,x)&=p(z_t\mid z_{t-1},x)\cdot\frac{p(z_{t-1}\mid x)}{p(z_t\mid x)}\\ &=N(\sqrt{\alpha_t}z_{t-1},\beta_t)\cdot \frac{N(\sqrt{\bar{\alpha}_{t-1}}x,1-\bar{\alpha}_{t-1}) }{N(\sqrt{\bar{\alpha}_t}x,1-\bar{\alpha}_t) }\\ &=N(\frac{\sqrt{\alpha_{t}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}} z_t+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1-\bar{\alpha}_{t}} x,\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \beta_{t}) \\ &=N(\mu_1,\sigma^{2}_1) \end{align} \tag{3.15}$$  
显然 $p(z_{t-1}\mid z_t,x)$ 是高斯分布，那么第三项就很容易算了：

$$\begin{align} KL(p(z_{t-1}\mid z_t,x)\|p_\theta(z_{t-1}\mid z_t)) &=\mathbb{E}_{p}\left[\log \frac{ \frac{1}{\sqrt{2\pi}\sigma_1}e^{-(z_{t-1}-\mu_1)^2/{2\sigma_{1}^2}} } {\frac{1}{\sqrt{2\pi}\sigma}e^{-(z_{t-1}-\mu)^2/{2\sigma^2}}} \right]\\ &=\mathbb{E}_{p}\left[\log\sigma - \log\sigma_{1} - (z_{t-1}-\mu_1)^2/{2\sigma^{2}_1} + (z_{t-1}-\mu)^2/{2\sigma^{2}} \right]\\ &=\log\sigma-\log\sigma_1-\frac{1}{2}+\frac{\sigma_{1}^2+(\mu_1-\mu)^2}{2\sigma^2} \end{align}\tag{3.16}$$  


由于 $\sigma$ 和 $\sigma_1$ 均为常量，所以实际上需要优化的就是 $\mu$ 和 $\mu_1$ 的MSE了，完成了扩散模型的推导。因此，扩散模型的损失函数 $\mathcal{L}$ 就为各个隐变量分布均值的MSE了。原论文实验来看，实际训练使用的是噪声，可能略有差别，但本质是完全一致的。

那么，为什么不像似然一样，直接用神经网络拟合所有的后验分布呢，而是要假设服从这么奇怪的高斯分布？个人的理解是，如果直接用神经网络拟合中间所有的过程，是非常难以训练的。因为对隐变量后验分布没有任何约束，中间过程是未知拟合未知的训练模式，会非常的不稳定，甚至退化到VAE。

总之和VAE相比，扩散模型的损失函数前两项是 $p(z_T\mid x)$和 $p(z_T)$ KL散度（作用是让 $x$ 对应的分布逼近先验）和似然。而第三项的实质是马尔科夫假设带来的约束，让隐变量像是用一根线串了起来。倘若没有这个假设，多个隐变量之间可能像一张网，相互影响，那么可以预想到第三项会十分复杂，甚至没有显示的数学表达式。

### 3.1.4 流模型 Flow-Based Models  
流模型是希望将 $x$ 和隐变量 $z$ 能够一一对应，有点类似AE的单隐变量模型，但不同的是，它还希望这个对应是双射的。即真实后验分布 $p(z\mid x)=\delta(z-C(x))$，真实似然 $p(x\mid z)=\delta(x-G(z))$ ，$z=C(x)$， $x=G(z)$ ，且 $C=G^{-1}$ 。假设先验分布 $p(z)=N(0,1)$ ，所以流模型是将 $p(x)$ 双射到 $p(z)$ 上，训练映射 $C_\theta$ ，这一点又和VAE很像。生成的时候采样 $z$ ，利用 $x=G_\theta(z)=C^{-1}_\theta(z)$ 就能完成，这就要求我们对编码器和解码器，也就是 $C$ 和 $G$ 的建模也需要可逆。既然要求可逆，首先 $x$ 和 $z$ 的维度要相同，便于推理，在本文都看做一维。接着观察真实分布 $p(x)$ ：

$$\begin{align} p(x) &=\int q(z)p(x\mid z)dz\\ &=\int q(z)\delta(x-G(z))dz\\ &=\int q(C(x))\delta(G(z)-x)\left| \frac{\partial C(x)}{\partial x} \right|dx &z=C(x)\\ &=q(C(G(z)))\left| \frac{\partial C(x)}{\partial x} \right|\\ &=q(C(x))\left| \frac{\partial C(x)}{\partial x} \right| \end{align}\tag{3.17}$$  
 其中 $\left| \frac{\partial C(x)}{\partial x} \right|$ 为雅克比行列式，且为了区分，将先验 $p(z)$ 写作成 $q(z)$ ，那么 $p_\theta(x)$ 的对数似然：

$$\begin{align} \mathcal{L} &=\log p_\theta(x)\\ &= \log q(C_\theta(x))\left| \frac{\partial C_\theta(x)}{\partial x} \right|\\ &=\log \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}C_{\theta}^2} +\log \left| \frac{\partial C_\theta(x)}{\partial x} \right|\\ &= \log \frac{1}{\sqrt{2\pi}}-\frac{1}{2}C_{\theta}^2(x)+\log \left| \frac{\partial C_\theta(x)}{\partial x} \right|\\ \end{align}\tag{3.18}$$  
所以在流模型中最大的问题是设计模型，使得雅克比行列式的值为1。最经典的是设计成放仿射模型，这样就能满足可逆的需求，这里就不做说明了。从某种程度来看，流模型也像是多隐变量模型，隐变量之间用的是特殊的结构（例如线性耦合）。

### 3.1.5 小结  
$p(x)$ 不方便直接拟合，那就将其拆成关于隐变量积分的形式，对先验、后验、似然进行建模，利用极大似然估计给定约束。

1. AE的思想是 $x$ 和 $z$ 唯一对应，VAE是希望 $x$ 和某个分布对应，他们的编码器和解码器都是直接用神经网络整个去拟合了；
2. Diffusion和VAE一样，也希望 $x$ 和某个分布对应，但是它给了编码器一个链状结构，而不是直接用一个神经网络整个去模拟了；
3. Flow流模型和AE类似，希望 $x$ 和 $z$ 唯一对应，但是它要求更高，还希望编码器和解码器互逆，于是在设计上比Diffusion的编码器结构更加严格。

### 3.2 基于能量函数的模型 Energy-Based Models  
考虑直接对 $p(x)$ 建模：

$$p_\theta(\bm{x})=\frac{q_\theta(\bm{x})}{Z(\theta)}\tag{3.19}$$  


$Z(\theta)=\int q_\theta(\bm{x}) d\bm{x}$ ，也叫作归一化因子；特别的，如果 $q_\theta(\bm{x})=e^{-U_\theta(\bm{x})}$ ：

$$p_\theta(\bm{x})=\frac{e^{-U_\theta(\bm{x})}}{Z(\theta)}\tag{3.20}$$  


其中 $U_\theta(\bm{x})$ 是未定的函数，也叫能量函数；此时 $p_\theta(\bm{x})$ 叫做能量分布，也叫作**麦克斯韦-玻尔兹曼分布（Maxwell–Boltzmann Distribution）**。

### 3.2.1 得分匹配 Score Matching[3]  
极大似然估计是需要极大化 $\log p_\theta(\bm{\bm{x}})$ ，由于存在 $Z(\theta)$ ，积分里面的一般会用神经网络拟合，所以不好估计。但是 $\nabla_\bm{x} \log p_\theta(\bm{x})$ 是很好估计的，因为 $\nabla_\bm{x} \log p_\theta(\bm{x})=\nabla_\bm{x} q_{\theta}(\bm{x}) - \nabla_\bm{x}\log Z(\theta)=\nabla_\bm{x} q_{\theta}(\bm{x})$ ，这样就不需要计算 $Z(\theta)$ 了。而估计 $\nabla_\bm{x} \log p_\theta(\bm{x})$ 的意义是：有些时候估计 $\log p(\bm{x})$ 的梯度也能了解数据分布，因为它们之间只差了一个放缩因子 $Z(\theta)$ ，有帮助了解数据分布 $p(\bm{x})$ 的形状和走势，在后续的一些论文里面也经常用到。

定义 $\nabla_\bm{x} \log p_\theta(\bm{x})=s_\theta(\bm{x})$ 为得分函数（score function）：

$$\nabla_\bm{x} \log p_\theta(\bm{x})= \begin{pmatrix} \frac{\partial\log p_\theta(\bm{x})}{\partial x_1} \\ \frac{\partial\log p_\theta(\bm{x})}{\partial x_2} \\ ...\\ \frac{\partial\log p_\theta(\bm{x})}{\partial x_n}\\ \end{pmatrix} = \begin{pmatrix} s_{\theta,1}(\bm{x}) \\ s_{\theta,2}(\bm{x}) \\ ...\\ s_{\theta,n}(\bm{x})\\ \end{pmatrix} =s_\theta(\bm{x})\tag{3.21}$$  
一般认为 $\bm{x}\in \mathbb{R}^n$ ，所以得分函数就是对 $\log p_\theta(\bm{x})$ 每个维度求偏导的向量。一般来说，我们用极大似然估计出来的目标是极小化MSE，但这里可以用Fisher散度来衡量分布之间的逼近程度，所以目标函数可表示为：

$$\begin{align} J(\theta) &=\frac{1}{2} \int p(\bm{x})\left\|\nabla_\bm{x} \log p_\theta(\bm{x})-\nabla_{\bm{x}} \log p(\bm{x})\right\|^{2} d\bm{x}\\ &=\frac{1}{2} \int p(\bm{x})\left\|s_\theta(\bm{x})-s(\bm{x})\right\|^{2} d\bm{x}\\ \end{align}\tag{3.22}$$  
 接着再对目标函数进行一些变化：

$$\begin{align} J(\theta)= &\frac{1}{2} \int p(\bm{x})\left\|s_\theta(\bm{x})-s(\bm{x})\right\|^{2} d\bm{x}\\ &=\int p(\bm{x}) \left[ \frac{1}{2}\left\| s_{\theta}(\bm{x})\right\|^2  - s_{\theta}^T(\bm{x}) s(\bm{x}) +\frac{1}{2}\left\|s(\bm{x}) \right\|^2 \right]d\bm{x} \end{align}\tag{3.23}$$  
第一项是模型得分函数各维度的平方和，能直接求出来；第三项与 $\theta$ 无关，可以直接忽略；关于第二项，对 $s(\bm{x})$ 第 $i$ 个元素上有：

$$\begin{align} - \int p(\bm{x})s_{\theta,i}(\bm{x}) s_i(\bm{x})d\bm{x} &=- \int p(\bm{x})s_{\theta,i}(\bm{x}) \frac{\partial \log p(\bm{x})}{\partial x_i}d\bm{x} \\ &=- \int s_{\theta,i}(\bm{x}) \frac{\partial p(\bm{x})}{\partial x_i}d\bm{x} \\ &=- \int... \left[\int s_{\theta,i}(\bm{x})\frac{\partial p(\bm{x})}{\partial x_i}dx_i \right] dx_1...dx_{i-1}dx_{i+1}...dx_n\\ &=- \int... \left[ s_{\theta,i}(\bm{x})p(\bm{x})\bigg|_{x_i=-\infty}^{x_i=+\infty}-\int p(\bm{x})\frac{\partial s_{\theta,i}(\bm{x})}{\partial x_i}dx_i  \right] dx_1...dx_{i-1}dx_{i+1}...dx_n\\ \end{align}\tag{3.24}$$  


因为可微的概率分布 $p(\bm{x})$ 是有界的，所以 $\lim_{x_i \rightarrow \infty}{p(\bm{x})}=0$ ，第一项为0。那么原式就变成了：

$$\begin{align} - \int p(\bm{x})s_{\theta,i}(\bm{x}) s_i(\bm{x})d\bm{x} &=\int p(\bm{x})\frac{\partial s_{\theta,i}(\bm{x})}{\partial x_i}d\bm{x} \\ \end{align}\tag{3.25}$$  


那么对于所有元素有：

$$\begin{align} - \int p(\bm{x})s_{\theta}^T(\bm{x}) s(\bm{x})d\bm{x} &=\int p(\bm{x}) \|\nabla_{\bm{x}} s_{\theta}(\bm{x})\|d\bm{x} \\ \end{align}\tag{3.26}$$  


最终：

$$\begin{align} J(\theta) &=\int p(\bm{x}) \left[ \frac{1}{2}\left\| s_{\theta}(\bm{x})\right\|^2  - s_{\theta}^T(\bm{x}) s(\bm{x}) \right]d\bm{x} + \text{const}\\ &=\int p(\bm{x})\left[ \frac{1}{2}\left\| s_{\theta}(\bm{x})\right\|^2 + \|\nabla_\bm{x} s_\theta(\bm{x})\|\right]d\bm{x}+ \text{const}\\ \end{align}\tag{3.27}$$  


可以看到，得分匹配算法实际上提供了对 $\nabla_{\bm{x}}\log p(\bm{x})$ 估计的一种方法，虽然没办法利用结果直接进行生成，但有助于我们了解复杂 $p(\bm{x})$ 的形状，方便我们研究它的其它性质。

### 3.2.2 生成对抗网络 Generative Adversarial Network[4][5]  
得分匹配选择避开了对 $p(\bm{x})$ 的直接估计，转而估计 $\nabla_{\bm{x}}\log p(\bm{x})$ 。而GAN则对能量分布模型直接估计：

$$\begin{align} \mathcal{L} &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ -\log p_\theta(\bm{x})\right]\\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ -\log \frac{e^{-U_\theta(\bm{x})}}{Z(\theta)}\right]\\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ U_\theta(\bm{x})+\log Z(\theta) \right]\\ \end{align}\tag{3.28}$$  


接下来利用梯度下降法优化目标，那么考虑 $\nabla_\theta\mathcal{L}$ ：

$$\begin{align} \nabla_\theta\mathcal{L} &= \mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})+\nabla_\theta\log Z(\theta) \right] \\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]+\frac{1}{Z(\theta)}\nabla_\theta Z(\theta)  \\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]+\frac{1}{Z(\theta)}\nabla_\theta \int e^{-U_\theta(\bm{x})}d\bm{x}  \\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]-\frac{1}{Z(\theta)} \int e^{-U_\theta(\bm{x})} \nabla_\theta U_\theta(\bm{x})d\bm{x}  \\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]- \int p_\theta(\bm{x}) \nabla_\theta U_\theta(\bm{x})d\bm{x}  \\ &=\mathbb{E}_{\bm{x}\sim p(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]- \mathbb{E}_{\bm{x}\sim p_\theta(\bm{x})}\left[ \nabla_\theta U_\theta(\bm{x})\right]  \\ \end{align}\tag{3.29}$$  


这样就把 $Z(\theta)$ 不好计算的问题解决了。在之前的隐变量模型中我们都是利用重整参数化技巧对高斯分布进行采样，但这里的 $p_\theta(\bm{x})$ 并不是高斯分布。所以这又带来了一个新的问题： $p_\theta(\bm{x})$ 不好采样。一般我们熟悉的采样过程是：给定 $p(z)$ 为标准高斯分布，令 $x=G_\varphi(z)$ ，那么 $p_\varphi(x\mid z)=\delta(x-G_\varphi(z))$ ，则 $x$ 为 $p_\varphi(x)$ 的一个采样点。所以如果 $p_\varphi(x)$ 逼近 $p_\theta(x)$ ，近似 $p_\varphi(x)=p_\theta(x)$ ，那就能利用这个过程对 $p_\theta(x)$ 采样了。此时的梯度就可以写为：$$\begin{align} \nabla_\theta\mathcal{L} & =\mathbb{E}_{x \sim p(x)}\left[\nabla_\theta U_{\theta}(x)\right]-\mathbb{E}_{x \sim p_\varphi(x)}\left[\nabla_\theta U_{\theta}(x)\right]\\ \end{align}\tag{3.30}$$  


所以实际上的优化目标就变成了：

$$\begin{align} \mathcal{L} & =\mathbb{E}_{x \sim p(x)}\left[ U_{\theta}(x)\right]-\mathbb{E}_{x \sim p_\varphi(x)}\left[ U_{\theta}(x)\right]\\ \end{align}\tag{3.31}$$  


为了满足这个要求，同样可以用最小化KL散度来实现，则：

$$\begin{align} KL(p_\varphi(x)\|p_\theta(x)) &=\int p_\varphi(x)\log\frac{p_\varphi(x)}{p_\theta(x)}dx\\ &=\int p_\varphi(x)\log p_\varphi(x) dx - \int p_\varphi(x)\log p_\theta(x)dx\\ &=-H_\varphi(X) - \mathbb{E}_{x\sim p_\varphi(x)} \left[ \log p_\theta(x)\right] \\ &=-H_\varphi(X) - \mathbb{E}_{x\sim p_\varphi(x)} \left[ \log \frac{e^{-U_\theta(x)}}{Z(\theta)}\right] \\ &=-H_\varphi(X) + \mathbb{E}_{x\sim p_\varphi(x)} \left[  U_\theta(x)\right] +\text{const} \\ \end{align}\tag{3.32}$$  
因为：

$$\begin{align} p_\varphi(x) &=\int p_\varphi(x\mid z)p(z)dz\\ &=\int \delta(x-G_\varphi(x))\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}dz \end{align}\tag{3.33}$$  
这个积分显然无法解出来，所以第一项 $-H_\varphi(X)$ 没法直接计算。不过可以利用公式：

$$H_\varphi(X)=I_\varphi(X;Z) + H_\varphi(X\mid Z)\tag{3.34}$$  
而根据公式3.7，可以推出 $H_\varphi(X\mid Z)=\text{const}$ ，那么最终 ：

$$KL(p_\varphi(x)\|p_\theta(x))=\mathbb{E}_{x\sim p_\varphi(x)} \left[  U_\theta(x)\right] -I_\varphi(X;Z) +\text{const}\tag{3.35}$$  


有一些文章已经对互信息量做过估计了，例如[#ref\_6](#ref\_6)，而这一项只是为模型添加了正则。总的来看，优化应应该写成以下形式：

$$\begin{align} \theta & =\underset{\theta}{\arg \min } \quad\mathbb{E}_{x \sim p(x)}\left[U_{\theta}(x)\right]-\mathbb{E}_{x \sim p_\varphi(x)}\left[U_{\theta}(x)\right]\\ \varphi& =\underset{\varphi}{\arg \min }\quad \mathbb{E}_{x \sim p_\varphi(x)}\left[U_{\theta}(x)\right]-I_{\varphi}(X;Z) \end{align}\tag{3.36}$$  
显然对 $\theta$ 的优化代表辨别器，对 $\varphi$ 的优化代表生成器，这就导出了GAN。

### 3.2.3 基于MCMC的EBM MCMC-Based EBM[7]  
实际上，我们可以由 $\nabla_{x}\log p_{\theta}(x)$ 直接对 $p_\theta(x)$ 直接采样，这是因为当 $\varepsilon \rightarrow 0$ ，朗之万方程（Langevin Equation）的稳态解为能量分布：

$$x_{t+1}=x_{t}-\frac{1}{2} \varepsilon \nabla_{x} U\left(x_{t}\right)+\sqrt{\varepsilon} \alpha, \quad \alpha \sim \mathcal{N}(0,1)\tag{3.37}$$  


也就是说，给定初试状态分布 $x_0$ ，通过公式3.37的迭代，最终得到的 $x_t$ 会服从：

$$p_\theta(x_t)=\frac{e^{-U_\theta(x_t)}}{Z(\theta)}\tag{3.38}$$  


这样就完成了对 $p_\theta(x)$ 的采样了。

### 3.2.4 小结  
对于生成模型，另外一种常规的建模是 $p_\theta(x)=\frac{e^{-U_\theta(x)}}{Z(\theta)}$ ：

1. 得分匹配实际对 $\nabla_x\log p_\theta(x)$ 做出估计，与 $p_\theta(x)$ 相差了一个放缩因子，有助于了解 $p_\theta(x)$ 的形状；
2. GAN选择直接计算 $p_\theta(x)$ 的极大似然，引入了 $p_\varphi(x)$ ，避免了对 $p_\theta(x)$ 的直接采样；
3. 基于MCMC的模型发现，能量分布是某一朗之万方程的稳态解，可以直接对 $p_\theta(x)$ 采样。

## 4 总结  
讨论了对 $p(x)$ 建模的几种方式，总结了各种基本生成模型的动机和思想，希望通过概率公式来引导对神经网络模型的假设和构建。

## 参考  
1. [#ref\_1\_0](#ref\_1\_0)Auto-encoding variational bayes [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
2. [#ref\_2\_0](#ref\_2\_0)Denoising Diffusion Probabilistic Models [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
3. [#ref\_3\_0](#ref\_3\_0)Estimation of Non-Normalized Statistical Models by Score Matching [https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
4. [#ref\_4\_0](#ref\_4\_0)Maximum Entropy Generators for Energy-Based Models [https://arxiv.org/abs/1901.08508](https://arxiv.org/abs/1901.08508)
5. [#ref\_5\_0](#ref\_5\_0)Maximum Entropy Generators for Energy-Based Models [https://arxiv.org/abs/1901.08508](https://arxiv.org/abs/1901.08508)
6. [#ref\_6\_0](#ref\_6\_0)Learning deep representations by mutual information estimation and maximization [https://arxiv.org/abs/1808.06670](https://arxiv.org/abs/1808.06670)
7. [#ref\_7\_0](#ref\_7\_0)Implicit Generation and Generalization in Energy-Based Models [https://arxiv.org/abs/1903.08689](https://arxiv.org/abs/1903.08689)
