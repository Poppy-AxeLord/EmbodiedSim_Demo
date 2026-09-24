#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专家协同波形扫描：找一条"能把球持续单向搓动"的经典协同
=========================================================

背景
----
03 工程里的行波协同用的是对称正弦 sin(w t + wave k)。对称波形每个周期内
"推"和"拉"的幅度相等，球被搓过去又被搓回来，所以**净转角很小**（实测 3 s 只有
约 51°，而累计转角有 103°）——它能演示"球在掌内转起来了"，但没法把球定向送到
某个角度。

要把它变成能用的专家（既能训模仿学习、又能当强化学习的对照基线），关键是
**波形非对称**：推程手指压得深（法向力大，摩擦能带动球），回程手指抬得浅
（法向力小，指尖相对球打滑，不把球带回来）。这正是真实手指步态（finger gaiting）
里"power stroke / return stroke"不对称的物理本质。

本脚本扫若干候选波形，用统一指标挑出最好的一条：
  * t_170 —— 净转角首次达到 170° 所需时间（越短越好，说明单向滚动能力强）
  * cumu  —— 3 s 内累计转角
  * 掉球率
"""
import math
import os
import sys

import numpy as np

import hand_env as E
import hand_common as H

HERE = os.path.dirname(os.path.abspath(__file__))

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------- 波形
def w_sin(th):
    return math.sin(th)


def w_skew_pos(th):
    return math.sin(th) + 0.30 * math.sin(2 * th)


def w_skew_neg(th):
    return math.sin(th) - 0.30 * math.sin(2 * th)


def w_skew_pos_big(th):
    return math.sin(th) + 0.55 * math.sin(2 * th)


def w_saw_up(th):
    return 2.0 * ((th / TWO_PI) % 1.0) - 1.0


def w_saw_down(th):
    return 1.0 - 2.0 * ((th / TWO_PI) % 1.0)


def w_half_sin(th):
    """只保留正半周、负半周压平 -> 单向推、无回拉。"""
    s = math.sin(th)
    return s if s > 0 else -0.15 * abs(s)


# ---------------------------------------------------------------- 变体表
# name, waveform, amp, freq, wave(指间相位差), j1_coef, j1_lag(rad)
VARIANTS = [
    ("基线 sin（03 原版）",        w_sin,          0.90, 1.2, 1.6, 0.30, 0.0),
    ("skew+0.30",                 w_skew_pos,     0.90, 1.2, 1.6, 0.30, 0.0),
    ("skew-0.30",                 w_skew_neg,     0.90, 1.2, 1.6, 0.30, 0.0),
    ("skew+0.55",                 w_skew_pos_big, 0.90, 1.2, 1.6, 0.30, 0.0),
    ("saw 上升锯齿",               w_saw_up,       0.90, 1.2, 1.6, 0.30, 0.0),
    ("saw 下降锯齿",               w_saw_down,     0.90, 1.2, 1.6, 0.30, 0.0),
    ("半波单向",                   w_half_sin,     0.90, 1.2, 1.6, 0.30, 0.0),
    ("skew+0.30 / J1 滞后 0.9",    w_skew_pos,     0.90, 1.2, 1.6, 0.45, 0.9),
    ("skew+0.30 / freq 2.0",      w_skew_pos,     0.90, 2.0, 1.6, 0.30, 0.0),
    ("skew+0.30 / wave 2.4",      w_skew_pos,     0.90, 1.2, 2.4, 0.30, 0.0),
    ("skew+0.30 / amp 1.0",       w_skew_pos,     1.00, 1.2, 1.6, 0.30, 0.0),
    ("skew+0.30 / freq1.2 wave0", w_skew_pos,     0.90, 1.2, 0.0, 0.30, 0.0),
]


def make_policy(env, wave_fn, amp, freq, wave, j1_coef, j1_lag):
    def pol(e):
        t = e._t * E.CTRL_DT
        q = e.q_grasp.copy()
        w = 2 * math.pi * freq
        for k, i0 in enumerate(H.IDX_FINGER_J4):
            th = w * t + wave * k
            s = wave_fn(th)
            q[i0 + 2] += amp * s
            q[i0 + 3] += amp * j1_coef * wave_fn(th - j1_lag)
        q[H.IDX_THJ1] += -0.35 * amp * math.sin(w * t)
        return np.clip((q - e.q_grasp) / E.ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    return pol


def evaluate(env, pol, n_ep=2, horizon=E.EPISODE_STEPS):
    """把目标设成 180°（实际到不了），从而关掉"提前成功终止"，单纯测滚动能力。"""
    t170s, cumus, nets, drops = [], [], [], 0
    for _ in range(n_ep):
        env.reset()
        env._theta_target = math.pi
        t170 = None
        while True:
            obs, r, term, trunc, info = env.step(pol(env))
            if t170 is None and info["net_rot_deg"] >= 170.0:
                t170 = info["step"] * E.CTRL_DT
            if term and info["ball_pos"][2] < E.DROP_Z:
                drops += 1
                break
            if trunc:
                break
        t170s.append(t170 if t170 is not None else float("inf"))
        cumus.append(info["cumu_rot_deg"])
        nets.append(info["net_rot_deg"])
    return dict(t170=float(np.mean(t170s)), cumu=float(np.mean(cumus)),
                net=float(np.mean(nets)), drops=drops, n=n_ep)


def main():
    env = E.ShadowHandReorientEnv(randomize=True)
    print(f"3 s 内累计转角 / 净转角基准 —— 目标区间 {math.degrees(E.TARGET_MIN):.0f}°"
          f"~{math.degrees(E.TARGET_MAX):.0f}°\n")
    print(f"{'变体':34s} {'t_170(s)':>9s} {'累计(°)':>9s} {'净(°)':>8s} {'掉球':>5s}")
    print("-" * 72)

    rows = []
    for v in VARIANTS:
        name, fn, amp, freq, wave, j1c, j1l = v
        pol = make_policy(env, fn, amp, freq, wave, j1c, j1l)
        r = evaluate(env, pol)
        rows.append((name, r))
        t170 = "未达到" if r["t170"] == float("inf") else f"{r['t170']:.2f}"
        print(f"{name:34s} {t170:>9s} {r['cumu']:9.1f} {r['net']:8.1f} {r['drops']:5d}")

    print("\n=== 排序（能到 170° 的优先，其次累计转角大）===")
    ok = [x for x in rows if x[1]["t170"] != float("inf")]
    ok.sort(key=lambda x: x[1]["t170"])
    rest = [x for x in rows if x[1]["t170"] == float("inf")]
    rest.sort(key=lambda x: -x[1]["cumu"])
    for i, (name, r) in enumerate(ok + rest, 1):
        t170 = "未达到" if r["t170"] == float("inf") else f"{r['t170']:.2f}s"
        print(f"  {i:2d}. {name:34s} t_170={t170:>8s} 累计={r['cumu']:7.1f}°")
    env.close()


if __name__ == "__main__":
    main()
