#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练吞吐剖析：时间到底花在环境上还是花在梯度上
==============================================

在 CPU 上跑 RL，最容易踩的两个坑都不是"环境太慢"：

  1. **梯度更新的占比被忽略**。PPO 每个 rollout 要跑
     n_epochs × (n_steps·n_envs / batch_size) 次反向传播。如果梯度项配得太重，
     环境明明能跑 2000 步/秒，实际训练只有两三百步/秒。
  2. **PyTorch 默认线程数和环境 worker 抢核**。SubprocVecEnv ×4 已经占了 4 个核，
     主进程里 torch 再开 4 个线程做反向传播，结果是两边都被拖慢。

这个脚本把一轮 rollout 和一次 train() 分开计时，直接算出"等效步/秒"，
用来定超参。**必须用文件方式运行**（SubprocVecEnv 用 spawn，
从 stdin 喂脚本时子进程无法重新导入 __main__，会直接挂住）。

用法
----
    python bench_speed.py
    python bench_speed.py --n-envs 4 --steps 4096
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def bench_one(n_envs=4, subproc=True, n_steps=1024, batch=256, n_epochs=10,
              threads=1, rounds=2, net=(256, 256), seed=0):
    torch.set_num_threads(threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from train_rl import make_vec

    venv = make_vec(n_envs, seed0=seed, subproc=subproc)
    model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=batch,
                n_epochs=n_epochs, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=list(net)))
    # collect_rollouts 依赖 learn() 事先搭好的内部状态（回调注册、last_obs、
    # ep_info_buffer 等）。SB3 把这一坨放在 _setup_learn() 里，直接复用它，
    # 比自己一个个赋值稳。
    class _Noop(BaseCallback):
        def _on_step(self) -> bool:
            return True

    n_rounds = rounds + 1
    cb = _Noop()
    model._setup_learn(total_timesteps=n_steps * n_envs * n_rounds, callback=cb)

    def roundtrip():
        # SB3 的 n_rollout_steps 是"每个环境走多少步"，一圈的总环境步数是
        # n_rollout_steps × n_envs。传错会越界写 rollout_buffer。
        model.collect_rollouts(model.env, cb, model.rollout_buffer,
                               n_rollout_steps=n_steps)

    roundtrip()                                              # 预热
    t_roll = t_train = 0.0
    for _ in range(rounds):
        t0 = time.perf_counter()
        roundtrip()
        t_roll += time.perf_counter() - t0
        t0 = time.perf_counter()
        model.train()
        t_train += time.perf_counter() - t0
    t_roll /= rounds
    t_train /= rounds
    venv.close()

    per_round = n_steps * n_envs
    return dict(n_envs=n_envs, subproc=subproc, n_steps=n_steps, batch=batch,
                n_epochs=n_epochs, threads=threads, net=list(net),
                per_round=per_round,
                rollout_s=t_roll, train_s=t_train,
                rollout_fps=per_round / t_roll,
                train_frac=t_train / (t_roll + t_train),
                effective_fps=per_round / (t_roll + t_train))


def main():
    ap = argparse.ArgumentParser(description="RL 训练吞吐剖析")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    cases = [
        dict(n_envs=4, subproc=True,  n_steps=1536, batch=256, n_epochs=10, threads=4),
        dict(n_envs=4, subproc=True,  n_steps=1536, batch=256, n_epochs=10, threads=1),
        dict(n_envs=4, subproc=True,  n_steps=2048, batch=512, n_epochs=4,  threads=1),
        dict(n_envs=4, subproc=True,  n_steps=3072, batch=512, n_epochs=3,  threads=1),
        dict(n_envs=1, subproc=False, n_steps=2048, batch=512, n_epochs=4,  threads=1),
    ]
    print("=" * 100)
    print("RL 训练吞吐剖析（rollout 与梯度更新分开计时）")
    print("=" * 100)
    print(f"{'n_envs':>6} {'vec':>8} {'n_steps':>8} {'batch':>6} {'epochs':>7} "
          f"{'thr':>4} | {'rollout s':>10} {'rollout fps':>12} {'train s':>9} "
          f"{'更新占比':>9} | {'等效 fps':>9}")
    print("-" * 100)

    rows, results = [], []
    for c in cases:
        r = bench_one(rounds=args.rounds, **c)
        results.append(r)
        print(f"{r['n_envs']:>6} {'subproc' if r['subproc'] else 'dummy':>8} "
              f"{r['n_steps']:>8} {r['batch']:>6} {r['n_epochs']:>7} "
              f"{r['threads']:>4} | {r['rollout_s']:>10.2f} "
              f"{r['rollout_fps']:>12.0f} {r['train_s']:>9.2f} "
              f"{r['train_frac'] * 100:>8.1f}% | {r['effective_fps']:>9.0f}",
              flush=True)

    best = max(results, key=lambda r: r["effective_fps"])
    print("-" * 100)
    print(f"最快组合：n_envs={best['n_envs']} n_steps={best['n_steps']} "
          f"batch={best['batch']} n_epochs={best['n_epochs']} "
          f"torch_threads={best['threads']} -> {best['effective_fps']:.0f} 步/秒")
    print(f"按此配置跑 600k 步约需 {600000 / best['effective_fps'] / 60:.1f} 分钟")

    os.makedirs(RESULTS, exist_ok=True)
    import json
    with open(os.path.join(RESULTS, "speed_bench.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=results, best=best), f, ensure_ascii=False, indent=2)
    print(f"明细 -> results/speed_bench.json")


if __name__ == "__main__":
    main()
