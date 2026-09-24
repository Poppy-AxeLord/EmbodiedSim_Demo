#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
采集模仿学习示范数据
====================

示范来源：`expert_modes.py` 里 9 套"四指行波 + 拇指反摆"参数族（**脚本专家**），
每集随机挑一套 + 随机行波初相位。

为什么用脚本专家而不是人遥操作
------------------------------
这个环境是无头运行的，没有可用的遥操作回路；而且本工程要对比的是
"多模态示范下 BC / ACT / DP 的行为差异"，需要**示范分布可控、可重复**。
脚本专家正好满足：同一套代码、同一个随机种子，任何人复现出来的数据完全一致。
机器人学习里用运动规划/脚本控制器造示范是常规做法。

数据格式（npz）
---------------
  obs        (N, 63) float32   每条示范的观测（动作之前的观测）
  act        (N, 24) float32   专家在该观测下执行的动作，∈[-1,1]
  ep_start   (K,)    int64     第 k 集在 obs 里的起始下标
  ep_len     (K,)    int64     第 k 集的长度
  ep_mode    (K,)    int64     该集用的是哪套策略
  ep_phase0  (K,)    float64   该集的行波初相位（隐变量，**不进观测**）
  ep_success (K,)    bool      该集是否达成任务
  ep_target  (K,)    float64   该集目标转角（弧度）
  ep_net     (K,)    float64   净转角（度）
  ep_cumu    (K,)    float64   累计转角（度）

用法
----
    python collect_demos.py --episodes 150 --tag v1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from hand_env import ShadowHandReorientEnv, CTRL_DT
from expert_modes import MODES, MODE_NAMES, N_MODES, mode_action

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "results")


def collect(episodes, seed0=0, tag="v1", randomize=True, max_steps=None,
            verbose=True):
    os.makedirs(DEMO_DIR, exist_ok=True)
    env = ShadowHandReorientEnv(randomize=randomize)
    rng = np.random.default_rng(seed0)

    obs_all, act_all = [], []
    ep_start, ep_len, ep_mode, ep_phase0 = [], [], [], []
    ep_succ, ep_target, ep_net, ep_cumu = [], [], [], []

    t0 = time.perf_counter()
    for ep in range(episodes):
        mode = int(rng.integers(N_MODES))
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))

        o_buf, a_buf = [], []
        while True:
            a = mode_action(env, env._t * CTRL_DT, mode, phase0)
            o_buf.append(obs)
            a_buf.append(a)
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc or (max_steps and env._t >= max_steps):
                break

        ep_start.append(len(obs_all))
        ep_len.append(len(o_buf))
        obs_all.extend(o_buf)
        act_all.extend(a_buf)
        ep_mode.append(mode)
        ep_phase0.append(phase0)
        ep_succ.append(bool(info["success"]))
        ep_target.append(float(info["theta_target"]))
        ep_net.append(float(info["net_rot_deg"]))
        ep_cumu.append(float(info["cumu_rot_deg"]))

        if verbose and (ep + 1) % 25 == 0:
            n_ok = int(np.sum(ep_succ))
            print(f"  {ep + 1:4d}/{episodes} 集 | 累计 {len(obs_all):,} 步 | "
                  f"成功 {n_ok}/{ep + 1} = {n_ok / (ep + 1) * 100:.0f}% | "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    env.close()
    obs = np.asarray(obs_all, np.float32)
    act = np.asarray(act_all, np.float32)
    data = dict(
        obs=obs, act=act,
        ep_start=np.asarray(ep_start, np.int64),
        ep_len=np.asarray(ep_len, np.int64),
        ep_mode=np.asarray(ep_mode, np.int64),
        ep_phase0=np.asarray(ep_phase0, np.float64),
        ep_success=np.asarray(ep_succ, bool),
        ep_target=np.asarray(ep_target, np.float64),
        ep_net=np.asarray(ep_net, np.float64),
        ep_cumu=np.asarray(ep_cumu, np.float64),
    )
    out = os.path.join(DEMO_DIR, f"demos_{tag}.npz")
    np.savez_compressed(out, **data)

    # ---------------- 诊断：证明"多模态"是数据里客观存在的
    diag = diagnose(data)
    diag["hop"] = "多模态度量"
    meta = dict(episodes=episodes, steps=int(obs.shape[0]), seed0=seed0,
                randomize=randomize, tag=tag, out=os.path.basename(out),
                modes=MODE_NAMES, **diag)
    with open(os.path.join(DEMO_DIR, f"demos_{tag}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n数据 -> {out}")
        print(f"  {obs.shape[0]:,} 步 / {episodes} 集 / "
              f"{obs.shape[0] / (time.perf_counter() - t0):.0f} steps/s")
        print(f"  示范成功率 {np.mean(ep_succ) * 100:.1f}%")
        print(f"  观测 {obs.shape} 动作 {act.shape}")
        print("  各模式使用次数与成功率：")
        for i, nm in enumerate(MODE_NAMES):
            m = data["ep_mode"] == i
            if m.sum():
                print(f"    {nm:16s} {int(m.sum()):3d} 集 | "
                      f"成功 {np.mean(data['ep_success'][m]) * 100:5.1f}% | "
                      f"累计转角 {np.mean(data['ep_cumu'][m]):6.1f}°")
        print(f"\n  [多模态] 近邻观测间的动作分歧度 = {diag['action_spread']:.3f} "
              f"（动作自身尺度 = {diag['action_scale']:.3f}）")
        print(f"  [多模态] 该分歧里可由模式解释的比例（簇间/簇内方差比）= "
              f"{diag['between_over_within']:.2f}")
        print(f"  [多模态] k-means(k=4) 轮廓系数 = {diag['silhouette']:.3f}")
    return out, data, meta


# ------------------------------------------------------------------ 诊断
def _nn_pairs(obs, exclude, k=8, sample=2000, seed=0):
    """在观测空间里找最近邻对，返回 (i, j)。

    两个坑，都踩过：

    1. **别自己算 n×n 距离矩阵**：sample=3000、obs 63 维就是 3000×3000×63 个
       float64 ≈ 4.5 GB，直接吃爆内存。用 sklearn 的 kd/ball tree。
    2. **必须排除时间上相邻的步**：示范是连续轨迹，i 与 i+1 的观测几乎一样、
       动作也几乎一样。若不排除，最近邻全是 (i, i+1)，动作分歧恒等于 0，
       多模态就被稀释没了。这里强制要求 |i − j| > exclude。
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    sd = obs.std(0) + 1e-6                       # 按维度标准化，避免大尺度维度主导距离
    X = obs[idx] / sd
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(obs / sd)
    dist, nbr = nn.kneighbors(X)
    jdx = np.empty(len(idx), np.int64)
    dsel = np.empty(len(idx), np.float64)
    for a in range(len(idx)):
        pick = -1
        for c in range(1, nbr.shape[1]):
            if abs(int(nbr[a, c]) - int(idx[a])) > exclude:
                pick = c
                break
        if pick < 0:                              # 兜底：真的没有非相邻邻居
            pick = 1
        jdx[a] = nbr[a, pick]
        dsel[a] = dist[a, pick]
    return idx, jdx, dsel


def diagnose(data, exclude=30, sample=2000, seed=0):
    """量化示范数据的多模态程度。

    操作性定义：找一批**观测上很接近、但时间上不相邻**的样本对，看它们对应的
    动作差多少。

      * 动作分歧 ≈ 动作自身的尺度  -> p(a|obs) 确实不集中，多模态成立，
        BC 的 MSE 会去平均两个模式，落在一个"谁都不是"的位置；
      * 动作分歧 ≪ 动作尺度        -> p(a|obs) 基本是单峰的，BC 不会吃亏。

    再把分歧**分解**成两类最近邻对：
      * 跨模式对（两条示范用了不同参数套数）-> 多模态贡献
      * 同模式对（同套参数、仅相位不同）    -> 多模态与噪声混合
    比值 between/within 越大，越说明多模态是主要矛盾而不是噪声。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    obs, act = data["obs"], data["act"]
    idx, jdx, dist = _nn_pairs(obs, exclude, sample=sample, seed=seed)
    a1, a2 = act[idx], act[jdx]

    action_scale = float(np.sqrt(((act - act.mean(0)) ** 2).sum(1).mean()))
    pair_gap = np.sqrt(((a1 - a2) ** 2).sum(1))
    action_spread = float(pair_gap.mean())

    ends = data["ep_start"] + data["ep_len"]
    m1 = data["ep_mode"][np.searchsorted(ends, idx, side="right")]
    m2 = data["ep_mode"][np.searchsorted(ends, jdx, side="right")]
    p1 = data["ep_phase0"][np.searchsorted(ends, idx, side="right")]
    p2 = data["ep_phase0"][np.searchsorted(ends, jdx, side="right")]

    same_ep = (np.searchsorted(ends, idx, side="right")
               == np.searchsorted(ends, jdx, side="right"))
    cross = (m1 != m2) & ~same_ep
    within = (m1 == m2) & ~same_ep
    between = float(pair_gap[cross].mean()) if cross.any() else float("nan")
    within_g = float(pair_gap[within].mean()) if within.any() else float("nan")
    ratio = (float(between / max(within_g, 1e-9))
             if np.isfinite(between) and np.isfinite(within_g) else float("nan"))

    sub = np.concatenate([a1, a2], 0)
    if sub.shape[0] >= 40 and len(np.unique(sub, axis=0)) >= 4:
        km = KMeans(n_clusters=4, n_init=4, random_state=seed).fit(sub)
        sil = float(silhouette_score(sub, km.labels_))
    else:
        sil = float("nan")

    return dict(nn_obs_dist=float(dist.mean()),
                nn_pairs=int(len(idx)),
                cross_mode_pairs=int(cross.sum()),
                within_mode_pairs=int(within.sum()),
                same_episode_pairs=int(same_ep.sum()),
                action_spread=action_spread,
                action_scale=action_scale,
                spread_ratio=float(action_spread / max(action_scale, 1e-9)),
                between_mode_action_gap=between,
                within_mode_action_gap=within_g,
                between_over_within=ratio,
                silhouette=sil)


def main():
    ap = argparse.ArgumentParser(description="采集掌内搓球模仿学习示范")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    print("=" * 66)
    print(f"采集示范 · {args.episodes} 集 · {N_MODES} 套脚本专家（随机切换）")
    print("=" * 66)
    collect(args.episodes, seed0=args.seed, tag=args.tag, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
