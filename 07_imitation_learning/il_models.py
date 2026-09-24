#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三种模仿学习策略，同一份示范数据、同一个评测协议
================================================

  BC      : 确定性 MLP，观测 -> 单步动作，MSE 损失
  ACT     : CVAE + 动作分块（H=8）+ 推理时取先验 z=0
  DP      : 条件 DDPM（cosine 调度，T=100）+ 动作分块，推理用 DDIM 20 步采样

三者的差别只在"怎么建模 p(a | obs)"：

  观测 o 在示范数据里对应多个合理动作（9 套专家 + 随机相位）。
  * BC 学的是条件均值 E[a|o]。两个模式各占一半时，它输出两者的**中点**——
    而中点往往不在任何一条可行轨迹上（手指位置互相矛盾），开环跑下去球就掉。
    这是模仿学习里最经典的失效模式，不是调参没调好。
  * ACT 用 CVAE 把"风格"隐变量 z 从数据里逼出来，把 p(a|o) 拆成
    ∫p(a|o,z)p(z)dz。推理时取 z=0（先验均值）——如果训练时 KL 权重压得好，
    z=0 会落在一个"平均风格"上，仍然是某种折中；但它的动作分块本身
    也提供了额外的时序信息。实测里它介于 BC 和 DP 之间，这个结论是诚实的。
  * DP 直接建模整个分布，采样时从噪声出发走一遍反向扩散，**互斥的模式
    不需要被平均**——每一步去噪只往某个模式收敛。所以它是唯一能在
    "多模态"上明显赢过 BC 的。

关于 ACT 的诚实标注
-------------------
真正的 ACT 用 Transformer 做骨干 + 指数加权时序集成。这里为了在 CPU 上
分钟级跑完，骨干简化为 MLP（输入是整段观测块展平后的向量，仍然保留了
"看到一整段历史"的能力），推理用 receding horizon 而非时序集成
——因为时序集成会把不同模式的预测**平均**掉，恰好抹掉我们要展示的差异。
所以本工程里它叫 **ACT-lite**。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK = 8                 # 动作块长度 H
OBS_DIM = 63
ACT_DIM = 24


# ==================================================================== 数据
def build_chunk_index(ep_start, ep_len, H=CHUNK, stride=1):
    """构造 (M, H) 的下标矩阵，每行是"落在同一集内"的连续 H 个步下标。"""
    rows = []
    for s, L in zip(ep_start, ep_len):
        if L < H:
            continue
        starts = np.arange(s, s + L - H + 1, stride)
        rows.append(starts[:, None] + np.arange(H)[None, :])
    return np.concatenate(rows, 0).astype(np.int64)


class ChunkDataset(torch.utils.data.Dataset):
    """按需从 (obs, act) 现场拼动作块，不预先把 N×H×D 全展开存内存。"""

    def __init__(self, obs, act, cidx):
        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.act = torch.as_tensor(act, dtype=torch.float32)
        self.cidx = torch.as_tensor(cidx, dtype=torch.long)

    def __len__(self):
        return self.cidx.shape[0]

    def __getitem__(self, i):
        idx = self.cidx[i]
        return self.obs[idx].reshape(-1), self.act[idx].reshape(-1)


# ==================================================================== 通用
def _mlp(dims, out_act=None, p_drop=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if p_drop:
                layers.append(nn.Dropout(p_drop))
        elif out_act is not None:
            layers.append(out_act)
    return nn.Sequential(*layers)


def sinusoidal(t, dim):
    """连续时间步 -> 正弦位置编码。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                         device=t.device) / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


# ==================================================================== BC
class BCPolicy(nn.Module):
    """确定性单步策略：学条件均值 E[a|o]。"""

    kind = "bc"

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = _mlp([obs_dim, hidden, hidden, act_dim])

    def forward(self, obs):
        return self.net(obs)

    def loss(self, obs_chunk, act_chunk):
        # 只用块里的第一步作监督（BC 是单步模型）
        a0 = act_chunk.reshape(-1, CHUNK, ACT_DIM)[:, 0]
        o0 = obs_chunk.reshape(-1, CHUNK, OBS_DIM)[:, 0]
        return F.mse_loss(self.forward(o0), a0)

    @torch.no_grad()
    def sample_chunks(self, obs_chunk, n=1, **kw):
        """确定性策略：无论采多少次都只有一个答案——这正是问题所在。

        返回形状统一成 (B, n, H, A)（把同一个动作在 H 步上重复），下游的
        重规划/可视化就不用为 BC 单独分叉了。
        """
        o0 = obs_chunk.reshape(-1, CHUNK, OBS_DIM)[:, 0]
        y = self.forward(o0)                                    # (B, A)
        return y[:, None, None, :].repeat(1, n, CHUNK, 1)        # (B, n, H, A)


# ==================================================================== ACT
class ACTLite(nn.Module):
    """CVAE 学动作块。推理取 z=0（先验均值）。"""

    kind = "act"

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, H=CHUNK,
                 z_dim=32, hidden=256):
        super().__init__()
        self.H, self.act_dim, self.z_dim = H, act_dim, z_dim
        self.enc = _mlp([H * obs_dim + H * act_dim, hidden, hidden, 2 * z_dim])
        self.dec = _mlp([H * obs_dim + z_dim, hidden, hidden, H * act_dim])

    def encode(self, obs_chunk, act_chunk):
        h = self.enc(torch.cat([obs_chunk, act_chunk], -1))
        return h.chunk(2, -1)

    def decode(self, obs_chunk, z):
        y = self.dec(torch.cat([obs_chunk, z], -1))
        return y.reshape(-1, self.H, self.act_dim)

    def loss(self, obs_chunk, act_chunk, kl_w=1.0):
        mu, logvar = self.encode(obs_chunk, act_chunk)
        eps = torch.randn_like(mu) * torch.exp(0.5 * logvar)
        rec = F.mse_loss(self.decode(obs_chunk, mu + eps), 
                         act_chunk.reshape(-1, self.H, self.act_dim))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + kl_w * kl, rec.detach(), kl.detach()

    @torch.no_grad()
    def sample_chunks(self, obs_chunk, n=1, z=None):
        if z is None:
            z = torch.zeros(obs_chunk.shape[0], self.z_dim, device=obs_chunk.device)
        y = self.decode(obs_chunk, z)                       # (B, H, act_dim)
        return y[:, None, :, :].repeat(1, n, 1, 1)          # (B, n, H, act_dim)


# ==================================================================== DP
class DiffusionPolicyLite(nn.Module):
    """条件 DDPM：对动作块加噪去噪；训练只随机取一个 t，推理走 DDIM。"""

    kind = "dp"

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, H=CHUNK,
                 T=100, hidden=256, time_dim=64):
        super().__init__()
        self.H, self.act_dim, self.T, self.time_dim = H, act_dim, T, time_dim
        self.net = _mlp([H * obs_dim + H * act_dim + time_dim, hidden, hidden,
                         H * act_dim])
        # cosine 噪声调度（Nichol & Dhariwal）。**这里踩过一个很隐蔽的坑**：
        # 照搬 DDPM 原文的线性 beta（1e-4 → 0.02），当 T 只有几十步时
        # abar(T) ≈ 0.60 —— 加噪过程根本没走到"接近纯噪声"，而采样却从标准
        # 正态出发，训练分布与采样分布对不上，生成的动作整片飘离示范
        # （实测：到最近示范动作的距离是示范内部典型间距的 200 多倍，
        #  而同一个网络在闭环里成功率反而是最高的，非常容易误判）。
        # cosine 调度保证 abar(T) ≈ 0，训练和采样才一致。
        xs = torch.linspace(0, T, T + 1)
        abar = torch.cos(((xs / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
        abar = abar / abar[0]
        betas = (1.0 - abar[1:] / abar[:-1]).clamp(1e-4, 0.5)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("abar", torch.cumprod(1.0 - betas, 0))

    def eps_hat(self, obs_chunk, a_t, t):
        h = torch.cat([obs_chunk, a_t, sinusoidal(t, self.time_dim)], -1)
        return self.net(h)

    def loss(self, obs_chunk, act_chunk):
        y = act_chunk.reshape(-1, self.H * self.act_dim)
        t = torch.randint(0, self.T, (y.shape[0],), device=y.device)
        ab = self.abar[t][:, None]
        eps = torch.randn_like(y)
        a_t = ab.sqrt() * y + (1 - ab).sqrt() * eps
        return F.mse_loss(self.eps_hat(obs_chunk, a_t, t), eps)

    @torch.no_grad()
    def sample_chunks(self, obs_chunk, n=1, steps=20, generator=None):
        """DDIM 采样（用 T 的内部时间轴的等距子序列，steps 步走完）。"""
        B = obs_chunk.shape[0]
        dev = obs_chunk.device
        ts = torch.linspace(self.T - 1, 0, steps).round().long().to(dev)
        a = torch.randn(B, n, self.H * self.act_dim, device=dev, generator=generator)
        for i, t in enumerate(ts):
            t_b = t.repeat(B * n)
            oc = obs_chunk[:, None, :].repeat(1, n, 1).reshape(B * n, -1)
            e = self.eps_hat(oc, a.reshape(B * n, -1), t_b).reshape(B, n, -1)
            ab = self.abar[t]
            a0 = ((a - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-3.0, 3.0)
            if i + 1 < len(ts):
                ab_next = self.abar[ts[i + 1]]
                a = ab_next.sqrt() * a0 + (1 - ab_next).sqrt() * e
            else:
                a = a0
        return a.reshape(B, n, self.H, self.act_dim)


# ==================================================================== 训练
def train_model(model, ds, epochs=60, batch=256, lr=1e-3, val_frac=0.1,
                seed=0, verbose=True, log_every=10):
    torch.manual_seed(seed)
    n_val = max(1, int(len(ds) * val_frac))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=g)
    dl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,
                                     num_workers=0, drop_last=True)
    dl_val = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False,
                                         num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"epoch": [], "train": [], "val": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for obs_c, act_c in dl:
            out = model.loss(obs_c, act_c)
            loss = out[0] if isinstance(out, tuple) else out
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        trl = tot / max(nb, 1)
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for ob, ac in dl_val:
                o = model.loss(ob, ac)
                vl += float((o[0] if isinstance(o, tuple) else o).detach())
                vn += 1
            vll = vl / max(vn, 1)
        hist["epoch"].append(ep)
        hist["train"].append(trl)
        hist["val"].append(vll)
        if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
            print(f"    epoch {ep:3d}/{epochs}  train {trl:.5f}  val {vll:.5f}",
                  flush=True)
    return hist


# ==================================================================== 推理包装
class ChunkPolicy:
    """把动作块模型包成"每步给一个动作"的策略，用 receding horizon 重规划。

    每 replan 步重新推一次，执行块里的前 replan 个动作。
    不用时序集成（把重叠预测加权平均），因为平均会**跨模式平均**，
    恰好抹掉本项目要展示的多模态差异。
    """

    def __init__(self, model, act_mean, act_std, H=CHUNK, replan=4,
                 steps=20, z=None, device="cpu", seed=0):
        self.net = model.to(device).eval()
        self.mean = torch.as_tensor(act_mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(act_std, dtype=torch.float32, device=device)
        self.H, self.replan, self.steps, self.device = H, replan, steps, device
        self.z = z
        self.base_seed = seed
        self.reset()

    def reset(self):
        self._queue = []
        self._ep_seed = self.base_seed
        self._gen = torch.Generator(device="cpu").manual_seed(self.base_seed)
        self._ep_seed += 1

    @torch.no_grad()
    def _plan(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        oc = o.reshape(1, -1).repeat(1, self.H).reshape(1, -1)
        with torch.no_grad():
            if self.net.kind == "bc":
                y = self.net.sample_chunks(oc, n=1)           # (1,1,H,A)
            elif self.net.kind == "act":
                z = (torch.zeros(1, self.net.z_dim, device=self.device)
                     if self.z is None else
                     torch.full((1, self.net.z_dim), float(self.z), device=self.device))
                y = self.net.sample_chunks(oc, n=1, z=z)
            else:
                y = self.net.sample_chunks(oc, n=1, steps=self.steps,
                                           generator=self._gen)
        y = y[0, 0]                                            # (H, A)
        y = y * self.std[None, :] + self.mean[None, :]
        return y.cpu().numpy()

    def __call__(self, env, obs):
        if not self._queue:
            chunk = self._plan(obs)
            self._queue = list(chunk[:self.replan])
        return self._queue.pop(0).astype(np.float32)


@torch.no_grad()
def sample_action_distribution(model, obs_seq, act_mean, act_std, n=128,
                               steps=20, seed=0, device="cpu"):
    """给定观测，从一个策略里采 n 个动作（用于多模态可视化）。

    BC 采 n 次也只有 1 个不同的答案；ACT 靠 z~N(0,I) 采；DP 靠不同噪声采。
    """
    model = model.to(device).eval()
    mean = torch.as_tensor(act_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(act_std, dtype=torch.float32, device=device)
    o = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
    oc = o.reshape(1, -1)[:, None, :].repeat(1, CHUNK, 1).reshape(1, -1)

    if model.kind == "bc":
        y = model.sample_chunks(oc, n=1)                        # (1,1,H,A)
        a = y[0, 0, 0][None, :].repeat(n, 1)                    # 确定性 -> n 个全同
    elif model.kind == "act":
        # 注意：必须采 n 个**不同**的 z，否则 n 个样本完全一样，
        # 多模态对比就成了自己骗自己
        z = torch.randn(n, model.z_dim, device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
        a = model.decode(oc.repeat(n, 1), z)[:, 0, :]           # 取块内第一步
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        y = model.sample_chunks(oc, n=n, steps=steps, generator=g)
        a = y[0, :, 0, :]
    return (a * std[None, :] + mean[None, :]).cpu().numpy()
