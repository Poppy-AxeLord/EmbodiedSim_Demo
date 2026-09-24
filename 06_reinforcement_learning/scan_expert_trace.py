#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专家净转角轨迹扫描：确认滚动是否单向、以及目标区间该定多宽
============================================================

scan_expert.py 已确认两件事：
  1. 对称正弦（03 原版）净转角只有 44°/3s，不能定向送球；
  2. 换成非对称波形 sin+0.30·sin2θ 并把频率提到 2.0 Hz，净转角升到 95.6°/3s。

本脚本回答剩下两个设计问题：
  * 净转角是否**单调增长**（若单调，说明是真单向滚动，不是来回晃）——
    只有单调，才能把"目标角度"当作可达指标；
  * 按这个能力，episode 该多长、目标区间该定多宽才既有挑战又能被专家完成。
"""
import math
import os

import numpy as np

import hand_env as E
import hand_common as H

HERE = os.path.dirname(os.path.abspath(__file__))


def skewed(th, k):
    return math.sin(th) + k * math.sin(2 * th)


def make_policy(env, amp, freq, wave, skew=0.30):
    def pol(e):
        t = e._t * E.CTRL_DT
        q = e.q_grasp.copy()
        w = 2 * math.pi * freq
        for k, i0 in enumerate(H.IDX_FINGER_J4):
            s = skewed(w * t + wave * k, skew)
            q[i0 + 2] += amp * s
            q[i0 + 3] += amp * 0.30 * s
        q[H.IDX_THJ1] += -0.35 * amp * math.sin(w * t)
        return np.clip((q - e.q_grasp) / E.ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    return pol


def trace(env, pol, steps=2500):
    """返回逐步的净转角序列（度）。"""
    env.reset()
    env._theta_target = math.pi          # 关掉提前成功终止
    out = [0.0]
    for _ in range(steps):
        _, _, term, trunc, info = env.step(pol(env))
        out.append(info["net_rot_deg"])
        if term or trunc:
            break
    return np.array(out)


def main():
    env = E.ShadowHandReorientEnv(randomize=True)
    combos = [(0.90, 1.6, 1.6), (0.90, 2.0, 1.6), (0.90, 2.4, 1.6),
              (0.90, 2.0, 1.2), (0.90, 2.0, 2.0), (1.00, 2.4, 1.6),
              (0.90, 3.0, 1.6)]

    print("净转角轨迹（度）—— 每 0.4 s 采一次样，共 5 s")
    print(f"{'amp/freq/wave':>18s} " + "".join(f"{t:>7.1f}s" for t in
                                               [0.4 * i for i in range(1, 13)]))
    print("-" * 106)
    best = None
    for amp, freq, wave in combos:
        pol = make_policy(env, amp, freq, wave)
        tr = trace(env, pol, steps=2500)
        samples = [tr[min(int(0.4 * i / E.CTRL_DT), len(tr) - 1)] for i in range(1, 13)]
        tag = f"{amp:.2f}/{freq:.1f}/{wave:.1f}"
        print(f"{tag:>18s} " + "".join(f"{s:7.1f}" for s in samples))
        # 单调性：净转角相邻增长的占比
        d = np.diff(tr)
        mono = float(np.mean(d >= -0.05))
        peak = float(tr.max())
        if best is None or peak > best[1]:
            best = (tag, peak, mono)

    print(f"\n峰值净转角最高：{best[0]}  →  {best[1]:.1f}°"
          f"（净转角单调增长步占比 {best[2] * 100:.1f}%）")

    # 给出"能到达各角度门槛"的时间
    print("\n=== 各阈值到达时间（s），用最优组合 ===")
    amp, freq, wave = 0.90, 2.0, 1.6
    pol = make_policy(env, amp, freq, wave)
    tr = trace(env, pol, steps=4000)
    for thr in (30, 60, 90, 120, 150, 170):
        idx = np.where(tr >= thr)[0]
        print(f"  {thr:3d}°  " + (f"{idx[0] * E.CTRL_DT:.2f} s" if len(idx) else "未达到"))
    env.close()


if __name__ == "__main__":
    main()
