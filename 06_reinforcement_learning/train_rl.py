#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强化学习训练：PPO / SAC 在掌内定向旋转任务上
=============================================

同一套环境、同一套观测与动作空间下训练 PPO 与 SAC，并把**经典行波协同专家**
放在同一条曲线上做参照线，回答一个问题：

    不用示范、只给奖励，策略网络能不能自己学会把球搓到指定角度？

为什么在 CPU 上选这个任务规模
------------------------------
MuJoCo 步进实测约 4300 steps/s（含 24 关节 PD 力矩控制），4 核可跑 4 个并行环境。
任务用低维状态观测（63 维，不用图像），策略网络 2×256 MLP，
因此 1~2M 步在 CPU 上是分钟级的，不需要 GPU。

指标
----
  * 成功率：在 10° 容差内**连续稳定保持 0.12 s**（防止"粗暴挥动偶然扫过目标角"）
  * 目标误差：episode 结束时球相对朝向与目标角的差（度）
  * 掉球率

用法
----
    python train_rl.py --algo ppo --steps 1500000
    python train_rl.py --algo sac --steps  400000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

# 关键：SubprocVecEnv 已经占满 4 个核，torch 再开多线程做反向传播只会互相抢核。
# bench_speed.py 实测 rollout 吞吐 957 -> 1222 步/秒（+28%）。
torch.set_num_threads(1)

from plot_style import (use_cjk, CLR_PRIMARY, CLR_EXPERT, CLR_ACC, CLR_WARN)
use_cjk()
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hand_env import ShadowHandReorientEnv, EPISODE_STEPS, CTRL_DT

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------- 环境工厂
def make_env(seed: int, randomize: bool = True):
    """模块级工厂函数（SubprocVecEnv 在 Windows 上要用 spawn，必须可导入）。"""
    def _init():
        env = ShadowHandReorientEnv(randomize=randomize)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec(n_envs: int, seed0: int = 0, subproc: bool = True):
    """构造向量环境。

    实测（4 核 CPU、本机）：
        单环境串行            1248 步/秒
        DummyVecEnv ×4        1567 步/秒（受 GIL 限制，几乎不叠加）
        SubprocVecEnv ×4      2043 步/秒（真并行，但每步要 pickle 63 维观测 +
                              24 维动作跨进程，IPC 开销吃掉了大部分核间收益）

    所以这里默认仍走 SubprocVecEnv（多出的 ~60% 是净赚），但会把**实际生效**的
    类型打出来——回退到 DummyVecEnv 时不再是静默降级。
    """
    factories = [make_env(seed0 + i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    if subproc:
        try:
            vec = SubprocVecEnv(factories, start_method="spawn")
            print(f"[vec] SubprocVecEnv ×{n_envs}（多进程真并行）")
            return vec
        except Exception as e:  # pragma: no cover
            print(f"[vec] SubprocVecEnv 不可用（{type(e).__name__}: {e}），回退 DummyVecEnv")
    print(f"[vec] DummyVecEnv ×{n_envs}（单进程，受 GIL 限制）")
    return DummyVecEnv(factories)


# ---------------------------------------------------------------- 评估
def evaluate(policy_fn, n_ep=10, seed=12345, randomize=True):
    """policy_fn(env, obs) -> action。

    回调签名里显式带出 env，是因为**专家策略是有状态的**：expert_action() 依赖
    env._t 生成行波相位。若闭包持有另一个从未 step 的环境实例，相位恒为 0，
    专家就会退化成静止常量动作，基线随之失真。
    """
    env = ShadowHandReorientEnv(randomize=randomize)
    succ, drops, errs, cumus, rets, steps = 0, 0, [], [], [], []
    for i in range(n_ep):
        obs, info = env.reset(seed=seed + i)
        total = 0.0
        while True:
            a = policy_fn(env, obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        succ += int(info["success"])
        drops += int(info["dropped"])
        errs.append(info["theta_err_deg"])
        cumus.append(info["cumu_rot_deg"])
        rets.append(total)
        steps.append(info["step"] * CTRL_DT)
    env.close()
    return dict(success_rate=succ / n_ep, drop_rate=drops / n_ep,
                mean_err_deg=float(np.mean(errs)), mean_cumu_deg=float(np.mean(cumus)),
                mean_return=float(np.mean(rets)), mean_duration_s=float(np.mean(steps)))


def expert_policy(env, obs):
    """经典行波协同，从正在步进的 env 读相位。"""
    return env.expert_action()


def expert_eval(n_ep=20, seed=777):
    return evaluate(expert_policy, n_ep=n_ep, seed=seed)


# ---------------------------------------------------------------- 回调
class MetricsCallback(BaseCallback):
    """每隔若干步评估一次，记录成功率 / 目标误差 / 掉球率。"""

    def __init__(self, eval_freq=20000, n_eval=8, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self._next_eval = eval_freq
        self.hist = []          # (steps, success_rate, mean_err, drop_rate)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            m = evaluate(self._policy, n_ep=self.n_eval, seed=1000)
            self.hist.append((self.num_timesteps, m["success_rate"],
                              m["mean_err_deg"], m["drop_rate"]))
            if self.verbose:
                print(f"  [eval] step={self.num_timesteps:>9,d} "
                      f"success={m['success_rate'] * 100:5.1f}% "
                      f"err={m['mean_err_deg']:6.2f}° drop={m['drop_rate'] * 100:4.1f}%",
                      flush=True)
        return True

    def _policy(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action).reshape(-1)


def random_eval(n_ep=6, seed=555):
    """均匀随机策略基线：用来证明任务不是"随便动动就能过"。"""
    rng = np.random.default_rng(0)
    return evaluate(lambda env, obs: rng.uniform(-1, 1, env.n_hand).astype(np.float32),
                    n_ep=n_ep, seed=seed)


def plot_curves(hist, expert, rnd, algo, out_png, n_eval):
    if not hist:
        print("[warn] 没有评估记录，跳过绘图")
        return
    steps = [h[0] for h in hist]
    succ = [h[1] * 100 for h in hist]
    err = [h[2] for h in hist]
    drop = [h[3] * 100 for h in hist]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax[0].plot(steps, succ, "-o", ms=3.5, lw=1.6, color=CLR_PRIMARY,
               label=f"{algo.upper()} 训练中策略")
    ax[0].axhline(expert["success_rate"] * 100, ls="--", lw=1.4, color=CLR_EXPERT,
                  label=f"经典行波协同专家 {expert['success_rate'] * 100:.0f}%")
    ax[0].axhline(rnd["success_rate"] * 100, ls=":", lw=1.4, color=CLR_WARN,
                  label=f"随机策略 {rnd['success_rate'] * 100:.0f}%")
    ax[0].set_title(f"成功率（每次评估 {n_eval} 个 episode）")
    ax[0].set_xlabel("训练步数")
    ax[0].set_ylabel("%")
    ax[0].set_ylim(-4, 104)
    ax[0].legend(fontsize=8.5, loc="lower right")

    ax[1].plot(steps, err, "-o", ms=3.5, lw=1.6, color=CLR_ACC)
    ax[1].axhline(expert["mean_err_deg"], ls="--", lw=1.4, color=CLR_EXPERT,
                  label=f"专家 {expert['mean_err_deg']:.1f}°")
    ax[1].axhline(rnd["mean_err_deg"], ls=":", lw=1.4, color=CLR_WARN,
                  label=f"随机 {rnd['mean_err_deg']:.1f}°")
    ax[1].set_title("结束时目标误差（越低越好）")
    ax[1].set_xlabel("训练步数")
    ax[1].set_ylabel("度")
    ax[1].legend(fontsize=8.5)

    ax[2].plot(steps, drop, "-o", ms=3.5, lw=1.6, color=CLR_WARN)
    ax[2].axhline(expert["drop_rate"] * 100, ls="--", lw=1.4, color=CLR_EXPERT,
                  label=f"专家 {expert['drop_rate'] * 100:.0f}%")
    ax[2].set_title("掉球率（失控指标）")
    ax[2].set_xlabel("训练步数")
    ax[2].set_ylabel("%")
    ax[2].legend(fontsize=8.5)

    fig.suptitle(f"{algo.upper()} · Shadow Hand 掌内定向旋转 · CPU 训练（无示范、纯奖励驱动）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"学习曲线 -> {out_png}")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="掌内定向旋转 · 强化学习训练")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=25000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or args.algo
    print("=" * 66)
    print(f"强化学习训练 · {args.algo.upper()} · {args.steps:,} 步 · {args.n_envs} 并行环境")
    print("=" * 66)

    # 参照基线：经典专家 + 均匀随机
    print("评估基线……")
    expert = expert_eval(n_ep=20)
    print(f"  经典专家：成功率 {expert['success_rate'] * 100:.0f}% | "
          f"目标误差 {expert['mean_err_deg']:.1f}° | 掉球 {expert['drop_rate'] * 100:.0f}%")
    rnd = random_eval(n_ep=10)
    print(f"  随机策略：成功率 {rnd['success_rate'] * 100:.0f}% | "
          f"目标误差 {rnd['mean_err_deg']:.1f}° | 掉球 {rnd['drop_rate'] * 100:.0f}%\n")

    venv = make_vec(args.n_envs, seed0=args.seed)
    print(f"环境已就绪：观测 {venv.observation_space.shape} 动作 {venv.action_space.shape}")

    # 超参由 bench_speed.py 实测选定：rollout 吞吐 ~1200 步/秒封顶（策略前向推理
    # 本身就要 ~0.4 ms/步），所以尽量压低"每次 rollout 的梯度更新量"。
    # n_steps=3072 × 4 环境 = 12288 步一个 rollout，每轮只做 24×3 = 72 次反向传播，
    # 梯度占比从 36% 降到 11%，等效吞吐 616 -> 1073 步/秒。
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", venv, seed=args.seed, verbose=0,
            n_steps=3072, batch_size=512, n_epochs=3,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
            learning_rate=3e-4, target_kl=0.05,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",
        )
    else:
        # SAC 是异策略，每个环境步都要做梯度更新，在 CPU 上这是主要开销。
        # 用 train_freq=4 把更新频率降到 1/4，换取可接受的墙钟时间。
        model = SAC(
            "MlpPolicy", venv, seed=args.seed, verbose=0,
            learning_rate=3e-4, buffer_size=200_000, learning_starts=5000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=4,
            gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",
        )

    cb = MetricsCallback(eval_freq=args.eval_freq, n_eval=6, verbose=1)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0
    print(f"\n训练完成：{args.steps:,} 步 / {wall:.0f} s "
          f"= {args.steps / max(wall, 1e-9):,.0f} steps/s")

    # 最终评估
    final = evaluate(cb._policy, n_ep=20, seed=999)
    print(f"\n最终策略：成功率 {final['success_rate'] * 100:.0f}% | "
          f"目标误差 {final['mean_err_deg']:.1f}° | 掉球 {final['drop_rate'] * 100:.0f}% | "
          f"平均累计转角 {final['mean_cumu_deg']:.0f}°")

    model_path = os.path.join(RESULTS, f"{tag}_shadow_hand")
    model.save(model_path)
    print(f"模型 -> {model_path}.zip")

    metrics = dict(algo=args.algo, steps=args.steps, n_envs=args.n_envs,
                   wall_s=wall, throughput=args.steps / max(wall, 1e-9),
                   expert=expert, random=rnd, final=final, history=cb.hist)
    with open(os.path.join(RESULTS, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_curves(cb.hist, expert, rnd, tag,
                os.path.join(RESULTS, f"{tag}_learning_curve.png"), cb.n_eval)
    venv.close()


if __name__ == "__main__":
    main()
