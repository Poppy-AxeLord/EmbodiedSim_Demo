#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练三种模仿学习策略（同一份示范、同一套超参）
==============================================

    python train_il.py --algo all --epochs 60
    python train_il.py --algo dp --epochs 80

产出
----
  results/il_<algo>.pt         模型权重 + 动作归一化统计
  results/il_train_log.json    三个模型的训练/验证损失曲线
  results/il_loss_curves.png   损失曲线图
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from plot_style import use_cjk, CLR_PRIMARY, CLR_ACC, CLR_WARN
use_cjk()
import matplotlib.pyplot as plt

from il_models import (CHUNK, OBS_DIM, ACT_DIM, BCPolicy, ACTLite,
                       DiffusionPolicyLite, ChunkDataset, build_chunk_index,
                       train_model)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_demos(tag="v1", only_success=True):
    z = np.load(os.path.join(RES, f"demos_{tag}.npz"))
    obs, act = z["obs"], z["act"]
    es, el, em = z["ep_start"], z["ep_len"], z["ep_success"]
    if not only_success:
        return obs, act, es, el
    obs_s, act_s, start_s, len_s = [], [], [], []
    cur = 0
    for s, L, ok in zip(es, el, em):
        if ok:
            obs_s.append(obs[s:s + L])
            act_s.append(act[s:s + L])
            start_s.append(cur)
            len_s.append(L)
            cur += L
    return (np.concatenate(obs_s, 0), np.concatenate(act_s, 0),
            np.asarray(start_s, np.int64), np.asarray(len_s, np.int64))


def build(algo, act_mean, act_std, seed=0):
    torch.manual_seed(seed)
    if algo == "bc":
        return BCPolicy()
    if algo == "act":
        return ACTLite()
    if algo == "dp":
        return DiffusionPolicyLite()
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser(description="训练 BC / ACT-lite / Diffusion Policy-lite")
    ap.add_argument("--algo", default="all", choices=["bc", "act", "dp", "all"])
    ap.add_argument("--demos", default="v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=2, help="动作块下采样步长")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-demos", action="store_true",
                    help="连失败的示范也一起训（默认只用成功的）")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    obs, act, es, el = load_demos(args.demos, only_success=not args.all_demos)
    print("=" * 66)
    print(f"模仿学习训练 · 示范 {obs.shape[0]:,} 步 / {len(es)} 集"
          f"（{'含失败示范' if args.all_demos else '仅成功示范'}）")
    print("=" * 66)

    act_mean = act.mean(0)
    act_std = act.std(0) + 1e-6
    norm_act = (act - act_mean) / act_std

    cidx = build_chunk_index(es, el, H=CHUNK, stride=args.stride)
    print(f"动作块 H={CHUNK}  训练样本 {len(cidx):,} 个块")
    ds = ChunkDataset(obs, norm_act, cidx)

    algos = ["bc", "act", "dp"] if args.algo == "all" else [args.algo]
    logs, summary = {}, {}

    from il_models import train_model as _train

    for algo in algos:
        print(f"\n[{algo.upper()}]")
        model = build(algo, act_mean, act_std, seed=args.seed)
        n_par = sum(p.numel() for p in model.parameters())
        t0 = time.perf_counter()
        hist = _train(model, ds, epochs=args.epochs, batch=args.batch,
                      lr=args.lr, seed=args.seed, verbose=True)
        wall = time.perf_counter() - t0
        torch.save(dict(state_dict=model.state_dict(), kind=algo, H=CHUNK,
                        obs_dim=OBS_DIM, act_dim=ACT_DIM,
                        act_mean=act_mean, act_std=act_std,
                        epochs=args.epochs, train_seconds=wall),
                   os.path.join(RES, f"il_{algo}.pt"))
        logs[algo] = hist
        summary[algo] = dict(params=int(n_par), train_seconds=wall,
                            final_train_loss=hist["train"][-1],
                            final_val_loss=hist["val"][-1])
        print(f"    参数 {n_par:,} | 用时 {wall:.0f}s | "
              f"val loss {hist['val'][-1]:.5f} -> results/il_{algo}.pt")

    with open(os.path.join(RES, "il_train_log.json"), "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, history=logs, epochs=args.epochs,
                       demos=args.demos, chunk=CHUNK, stride=args.stride,
                       only_success=not args.all_demos, seed=args.seed),
                  f, ensure_ascii=False, indent=2)

    # ---------------- 损失曲线
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = {"bc": CLR_PRIMARY, "act": CLR_ACC, "dp": CLR_WARN}
    for algo in algos:
        h = logs[algo]
        ax[0].plot(h["epoch"], h["train"], lw=1.7, color=colors[algo],
                   label=f"{algo.upper()} 训练")
        ax[0].plot(h["epoch"], h["val"], lw=1.2, ls="--", color=colors[algo],
                   label=f"{algo.upper()} 验证")
        ax[1].plot(h["epoch"], h["val"], lw=1.7, color=colors[algo],
                   label=algo.upper())
    ax[0].set_title("训练 / 验证损失")
    ax[0].set_xlabel("epoch")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[1].set_title("验证损失对比（对数轴）")
    ax[1].set_xlabel("epoch")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=9)
    fig.suptitle("BC / ACT-lite / Diffusion Policy-lite · 同一示范数据集", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "il_loss_curves.png"))
    plt.close(fig)
    print("\n损失曲线 -> results/il_loss_curves.png")


if __name__ == "__main__":
    main()
