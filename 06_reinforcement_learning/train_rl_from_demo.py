#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从示范热启动的 RL（BC → PPO）
=============================

背景对比
--------
06 的 `train_rl.py` 是"从零开始、只给奖励"的对照：在本次 CPU 预算内（几十万步），
PPO 学会了稳住球（掉球率 50% → 0%），但**没学会"搓转"这个精细时序动作**
（成功率始终 0%，误差 ≈ 目标角本身，等价于握稳不动）。

这恰好说明了一个工程判断：接触富集的精细操作，纯 RL 从零学需要远超本次的算力。
真实项目里更常见的做法是**用示范给 RL 一个起点**——本脚本就做这件事。

实现
----
07 的 BC 是一个 63→256→256→24 的 MLP；SB3 的 ActorCriticPolicy
（net_arch=[256,256]）里 `mlp_extractor.policy_net` 也是 63→256→256，
`action_net` 是 256→24。**两者逐层形状完全一致**，所以可以直接把 BC 的权重搬过去，
不需要任何对齐技巧。

    policy_net[0] ← bc.net[0]     policy_net[2] ← bc.net[2]
    action_net    ← bc.net[4]

然后以较小的学习率继续 PPO 训练。注意这里**不是**"把 BC 当最优解"，而是把策略
初始化到"已经能做对动作"的区域，让 RL 去改进它 —— 这是 imitation + RL 的标准组合。

用法
----
    python train_rl_from_demo.py --steps 300000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)          # 与 train_rl.py 同理：别和 4 个环境 worker 抢核

from plot_style import use_cjk, CLR_PRIMARY, CLR_EXPERT, CLR_ACC, CLR_WARN
use_cjk()
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import train_rl as T
from hand_env import ShadowHandReorientEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BC_PATH = os.path.join(HERE, "..", "07_imitation_learning", "results", "il_bc.pt")


def inject_bc_weights(model, bc_path):
    """把 BC 的 MLP 权重灌进 PPO 的 Actor。返回是否成功。"""
    if not os.path.exists(bc_path):
        print(f"[warn] 找不到 {bc_path}，跳过热启动（等价于从零学）")
        return False
    ck = torch.load(bc_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    pol = model.policy
    mapping = [
        (pol.mlp_extractor.policy_net[0].weight, pol.mlp_extractor.policy_net[0].bias,
         sd["net.0.weight"], sd["net.0.bias"]),
        (pol.mlp_extractor.policy_net[2].weight, pol.mlp_extractor.policy_net[2].bias,
         sd["net.2.weight"], sd["net.2.bias"]),
        (pol.action_net.weight, pol.action_net.bias,
         sd["net.4.weight"], sd["net.4.bias"]),
    ]
    for (w, b, sw, sb) in mapping:
        assert w.shape == sw.shape and b.shape == sb.shape, \
            f"形状不匹配 {w.shape} vs {sw.shape} —— 网络结构变了"
        with torch.no_grad():
            w.copy_(sw)
            b.copy_(sb)
    print("已把 BC 权重注入 PPO 的 Actor（policy_net 两层 + action_net）")
    return True


def main():
    ap = argparse.ArgumentParser(description="BC 热启动的 PPO")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="热启动后用更小的学习率")
    ap.add_argument("--eval-freq", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ppo_warm")
    ap.add_argument("--no-bc", action="store_true", help="不注入 BC 权重（做消融对照）")
    args = ap.parse_args()

    print("=" * 66)
    print(f"BC 热启动 PPO · {args.steps:,} 步 · {args.n_envs} 并行环境"
          f"{' · [消融] 不注入 BC' if args.no_bc else ''}")
    print("=" * 66)

    expert = T.expert_eval(n_ep=20)
    rnd = T.random_eval(n_ep=10)
    print(f"  经典专家 成功率 {expert['success_rate'] * 100:.0f}% | "
          f"误差 {expert['mean_err_deg']:.1f}°")
    print(f"  随机策略 成功率 {rnd['success_rate'] * 100:.0f}% | "
          f"误差 {rnd['mean_err_deg']:.1f}°\n")

    venv = T.make_vec(args.n_envs, seed0=args.seed)
    model = PPO(
        "MlpPolicy", venv, seed=args.seed, verbose=0,
        n_steps=3072, batch_size=512, n_epochs=3,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.002, vf_coef=0.5, max_grad_norm=0.5,
        learning_rate=args.lr, target_kl=0.05,
        policy_kwargs=dict(net_arch=[256, 256]), device="cpu")

    if not args.no_bc:
        inject_bc_weights(model, BC_PATH)

    cb = T.MetricsCallback(eval_freq=args.eval_freq, n_eval=6, verbose=1)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0
    print(f"\n训练完成：{args.steps:,} 步 / {wall:.0f} s "
          f"= {args.steps / max(wall, 1e-9):,.0f} steps/s")

    # 热启动前的起点评估（用同一个回调口径，便于对比）
    final = T.evaluate(cb._policy, n_ep=20, seed=999)
    print(f"最终策略：成功率 {final['success_rate'] * 100:.0f}% | "
          f"误差 {final['mean_err_deg']:.1f}° | 掉球 {final['drop_rate'] * 100:.0f}% | "
          f"累计转角 {final['mean_cumu_deg']:.0f}°")

    model.save(os.path.join(RES, f"{args.tag}_shadow_hand"))
    with open(os.path.join(RES, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(steps=args.steps, lr=args.lr, wall_s=wall,
                       throughput=args.steps / max(wall, 1e-9),
                       bc_injected=not args.no_bc,
                       expert=expert, random=rnd, final=final, history=cb.hist),
                  f, ensure_ascii=False, indent=2)

    # ---------------- 曲线（与从零学的那条放在一起看才有意义）
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    zero_path = os.path.join(RES, "ppo_metrics.json")
    if os.path.exists(zero_path):
        z = json.load(open(zero_path, encoding="utf-8"))["history"]
        ax[0].plot([h[0] for h in z], [h[1] * 100 for h in z], "-o", ms=3,
                   lw=1.6, color=CLR_WARN, label="PPO 从零学")
        ax[1].plot([h[0] for h in z], [h[2] for h in z], "-o", ms=3, lw=1.6,
                   color=CLR_WARN, label="PPO 从零学")
    h = cb.hist
    ax[0].plot([x[0] for x in h], [x[1] * 100 for x in h], "-o", ms=3, lw=1.9,
               color=CLR_ACC, label="PPO（BC 热启动）")
    ax[1].plot([x[0] for x in h], [x[2] for x in h], "-o", ms=3, lw=1.9,
               color=CLR_ACC, label="PPO（BC 热启动）")
    ax[0].axhline(expert["success_rate"] * 100, ls="--", lw=1.4, color=CLR_EXPERT,
                  label=f"经典专家 {expert['success_rate'] * 100:.0f}%")
    ax[1].axhline(expert["mean_err_deg"], ls="--", lw=1.4, color=CLR_EXPERT,
                  label=f"专家 {expert['mean_err_deg']:.1f}°")
    ax[0].set_title("成功率")
    ax[0].set_ylabel("%")
    ax[0].set_ylim(-4, 104)
    ax[1].set_title("结束时目标误差")
    ax[1].set_ylabel("度")
    for a in ax:
        a.set_xlabel("训练步数")
        a.legend(fontsize=8.5)
    fig.suptitle("示范热启动 vs 从零学：同一环境、同一奖励、同一网络", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"{args.tag}_learning_curve.png"))
    plt.close(fig)
    print(f"学习曲线 -> results/{args.tag}_learning_curve.png")
    venv.close()


if __name__ == "__main__":
    main()
