#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 导出 + 推理延迟基准
========================

训练出来的策略要真的上车/上机，就得离开 PyTorch 运行时。ONNX 是这条路上
最通用的一站：导出成图，用 onnxruntime 跑，不依赖 Python 侧的 autograd 开销。

本脚本做三件事，缺一不可：

1. **导出**：把 06 训练好的 PPO 策略切成"纯 Actor"（观测 -> 确定性动作），
   导出 ONNX。注意必须切掉 value head 和 log_std —— 部署时只用均值动作。
2. **一致性验证**：256 条随机观测上对齐 PyTorch 输出，报最大绝对偏差。
   只报"导出成功"是不够的，**必须证明数值等价**，否则导出的可能是错的图。
3. **延迟基准**：单样本延迟（p50 / p95）与批量吞吐，对比
   PyTorch(1 线程) / PyTorch(4 线程) / onnxruntime(1 线程) / onnxruntime(默认)。
   控制回路是 500 Hz（2 ms），所以**单样本延迟**才是关键指标，批量吞吐只是参考。

用法
----
    python deploy_onnx.py --model ../06_reinforcement_learning/results/ppo_shadow_hand.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from plot_style import use_cjk, CLR_PRIMARY, CLR_ACC, CLR_WARN, CLR_EXPERT, CLR_GRAY
use_cjk()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


class ActorOnly(nn.Module):
    """只保留确定性 Actor：观测 -> 动作均值。

    SB3 的 ActorCriticPolicy 里同时有 value head 和 log_std，部署时用不到。
    这里只取 policy_net + action_net，**连 value_net 都不挂上来**，
    否则参数量统计会把没用的价值网络也算进去。
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.register_buffer("log_std_ignored",
                             policy.log_std.detach().clone(), persistent=False)

    def forward(self, obs):
        return self.action_net(self.policy_net(obs))


def latency(fn, n_warm=50, n_run=500):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return dict(p50=float(np.median(ts)), p95=float(np.percentile(ts, 95)),
                mean=float(ts.mean()))


def main():
    ap = argparse.ArgumentParser(description="ONNX 导出与推理延迟基准")
    ap.add_argument("--model", default=os.path.join(
        HERE, "..", "06_reinforcement_learning", "results", "ppo_shadow_hand.zip"))
    ap.add_argument("--obs-dim", type=int, default=63)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    from stable_baselines3 import PPO
    import onnxruntime as ort

    print("=" * 70)
    print("ONNX 导出与推理延迟基准")
    print("=" * 70)
    if not os.path.exists(args.model):
        raise SystemExit(f"找不到策略文件：{args.model}\n先跑 06 的 train_rl.py")

    # 把策略复制进本项目的 results，保持子项目自包含
    local_zip = os.path.join(RES, "ppo_shadow_hand.zip")
    if os.path.abspath(args.model) != os.path.abspath(local_zip):
        shutil.copy2(args.model, local_zip)
        print(f"策略已复制 -> results/ppo_shadow_hand.zip")

    model = PPO.load(local_zip, device="cpu")
    actor = ActorOnly(model.policy).eval()
    n_par = sum(p.numel() for p in actor.parameters())
    print(f"Actor 参数量 {n_par:,}")

    # ---------------- 导出
    onnx_path = os.path.join(RES, "policy.onnx")
    dummy = torch.zeros(1, args.obs_dim)
    # 不用 dynamo 导出路径：torch>=2.9 默认走 dynamo，会依赖额外的 onnxscript 包。
    # 这里网络就是几层 MLP，TorchScript 导出器完全够用、且没有额外依赖。
    torch.onnx.export(
        actor, dummy, onnx_path,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"ONNX -> {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

    # ---------------- 一致性验证
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, (256, args.obs_dim)).astype(np.float32)
    with torch.no_grad():
        y_torch = actor(torch.as_tensor(obs)).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"obs": obs})[0]
    diff = float(np.abs(y_torch - y_ort).max())
    rel = float(np.abs(y_torch - y_ort).max() / (np.abs(y_torch).max() + 1e-12))
    print(f"\n一致性：256 条观测上最大绝对偏差 {diff:.3e}"
          f"（相对 {rel:.3e}）-> {'通过' if diff < 1e-4 else '偏差过大，需检查'}")

    # ---------------- 延迟
    torch.set_num_threads(1)
    x1 = torch.as_tensor(obs[:1])
    with torch.no_grad():
        lat_t1 = latency(lambda: actor(x1))
    torch.set_num_threads(4)
    lat_t4 = latency(lambda: actor(x1))
    torch.set_num_threads(1)

    so1 = ort.SessionOptions()
    so1.intra_op_num_threads = 1
    so1.inter_op_num_threads = 1
    s1 = ort.InferenceSession(onnx_path, so1, providers=["CPUExecutionProvider"])
    lat_o1 = latency(lambda: s1.run(None, {"obs": obs[:1]}))
    lat_od = latency(lambda: sess.run(None, {"obs": obs[:1]}))

    rows = [("PyTorch (1 线程)", lat_t1, CLR_PRIMARY),
            ("PyTorch (4 线程)", lat_t4, CLR_WARN),
            ("onnxruntime (1 线程)", lat_o1, CLR_ACC),
            ("onnxruntime (默认)", lat_od, CLR_EXPERT)]
    print(f"\n单样本推理延迟（obs = 1×63）")
    for name, m, _c in rows:
        print(f"  {name:22s} p50 {m['p50']:6.2f} ms | p95 {m['p95']:6.2f} ms | "
              f"≈{1000 / max(m['p50'], 1e-9):6.0f} Hz")
    print("  控制周期 2 ms（500 Hz）—— 只有 p50 明显低于 2 ms 才谈得上实时。")

    # 批量吞吐
    batch = {}
    for b in (1, 64, 1024):
        xb = np.repeat(obs[:1], b, axis=0)
        tb = torch.as_tensor(xb)
        torch.set_num_threads(4)
        with torch.no_grad():
            actor(tb)
            t0 = time.perf_counter()
            for _ in range(20):
                actor(tb)
            tth = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run(None, {"obs": xb})
        tor = (time.perf_counter() - t0) / 20
        batch[b] = dict(torch_per_s=b / tth, ort_per_s=b / tor)
    torch.set_num_threads(1)
    print("\n批量吞吐（样本/秒）")
    for b, v in batch.items():
        print(f"  batch {b:5d} | PyTorch {v['torch_per_s']:12,.0f} | "
              f"onnxruntime {v['ort_per_s']:12,.0f}")

    # ---------------- 图
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    names = [r[0] for r in rows]
    p50 = [r[1]["p50"] for r in rows]
    p95 = [r[1]["p95"] for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.18, p50, width=0.36, color=CLR_ACC, label="p50")
    ax[0].bar(x + 0.18, p95, width=0.36, color=CLR_WARN, label="p95")
    ax[0].axhline(2.0, ls="--", lw=1.4, color=CLR_EXPERT, label="控制周期 2 ms")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=16, ha="right", fontsize=8.5)
    ax[0].set_ylabel("ms")
    ax[0].set_title("单样本推理延迟")
    ax[0].legend(fontsize=8.5)
    ax[0].margins(y=0.2)

    ax[1].scatter(y_torch.ravel(), y_ort.ravel(), s=6, c=CLR_PRIMARY, alpha=0.5)
    lim = [float(min(y_torch.min(), y_ort.min())), float(max(y_torch.max(), y_ort.max()))]
    ax[1].plot(lim, lim, ls="--", lw=1.2, color=CLR_EXPERT)
    ax[1].set_xlabel("PyTorch 动作输出")
    ax[1].set_ylabel("ONNX Runtime 输出")
    ax[1].set_title(f"数值一致性（最大偏差 {diff:.1e}）")

    bs = list(batch.keys())
    ax[2].plot(bs, [batch[b]["torch_per_s"] for b in bs], "-o", lw=1.7,
               color=CLR_PRIMARY, label="PyTorch (4 线程)")
    ax[2].plot(bs, [batch[b]["ort_per_s"] for b in bs], "-s", lw=1.7,
               color=CLR_ACC, label="onnxruntime (默认)")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("batch size")
    ax[2].set_ylabel("样本/秒")
    ax[2].set_title("批量吞吐")
    ax[2].legend(fontsize=9)

    fig.suptitle("PPO 策略 ONNX 导出 · 一致性验证与部署延迟基准", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "onnx_benchmark.png"))
    plt.close(fig)
    print("\n图 -> results/onnx_benchmark.png")

    with open(os.path.join(RES, "onnx_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(dict(params=int(n_par), onnx_kb=os.path.getsize(onnx_path) / 1024,
                       max_abs_diff=diff, max_rel_diff=rel,
                       latency={n: m for n, m, _ in rows},
                       batch_throughput=batch,
                       ctrl_period_ms=2.0),
                  f, ensure_ascii=False, indent=2)
    print("指标 -> results/onnx_benchmark.json")


if __name__ == "__main__":
    main()
