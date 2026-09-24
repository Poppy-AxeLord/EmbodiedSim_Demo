"""示范策略库：9 套**互相独立、都能成功**的掌内搓球策略。

为什么需要"多套"而不是一套
--------------------------
模仿学习里最经典的一个对照是：

    当示范数据本身是多模态的（同一观测对应多种合理动作），
    "对动作取平均"的 BC 会落在两个模式的中间——那个位置谁都不是，
    开环执行下去球就掉了；而 Diffusion Policy 因为建模的是整个
    动作分布，能采到具体的模式。

要做这个对照，示范数据的多模态必须是**真实存在、可测量**的，不能靠嘴说。
所以这里准备 9 组参数不同的"四指行波 + 拇指反摆"策略，它们：

  * 都来自 03 工程那条协同波形，只是幅度 / 频率 / 波数 / 不对称度 / 拇指反摆
    系数 / J1 耦合系数不同；
  * 实测都能把球搓到目标角（见 results/expert_modes.json，成功率 6~7/8）；
  * 动作统计差异很大——累计转角从 76°（H 无 J1 耦合，最省）到 173°
    （F 强不对称，最费），也就是说它们走的是**很不一样的动作轨迹**。

再加上每一集随机化初始相位 phase0，同一个 theta_now 就可以在很不同的手指
相位下出现，"同一观测 -> 多种动作"这件事就成了数据里的客观事实。

关于多模态来源的一句诚实说明
----------------------------
本任务的观测里有 theta_now 和 theta_target，但**没有**专家内部的行波相位
（那是专家的隐私状态）。因此相位不同的两条轨迹在观测上可以几乎重合、
动作却完全不同。这正是 p(a|obs) 多峰的物理来源。

（顺带排除一个想当然的设计：我们试过用反向行波 direction=-1 造"反向旋转"
二模态，实测**不成立**——反向行波因为 stick-slip 不对称仍然产生正向净转角，
只是变弱（29.3°/8 次成功 0 次）。所以多模态靠的是参数族 + 相位，不是方向。）
"""
from __future__ import annotations

import math

import numpy as np

import hand_common as H
from hand_env import ACTION_SCALE

# (名称, amp, freq, wave, skew, thumb_coef, j1_couple)
MODES = [
    ("A 基准协同",       1.00, 2.4, 1.6, 0.30, -0.35, 0.30),
    ("B 低幅",           0.80, 2.4, 1.6, 0.30, -0.35, 0.30),
    ("C 低频",           1.00, 1.9, 1.6, 0.30, -0.35, 0.30),
    ("D 高频",           1.00, 3.0, 1.6, 0.30, -0.35, 0.30),
    ("E 弱波数",         1.00, 2.4, 0.9, 0.30, -0.35, 0.30),
    ("F 强不对称",       1.00, 2.4, 1.6, 0.45, -0.35, 0.30),
    ("G 拇指强反摆",     1.00, 2.4, 1.6, 0.30, -0.70, 0.30),
    ("H 无 J1 耦合",     1.00, 2.4, 1.6, 0.30, -0.35, 0.00),
    ("I 高幅低频",       1.15, 1.9, 1.6, 0.30, -0.35, 0.30),
]

MODE_NAMES = [m[0] for m in MODES]
N_MODES = len(MODES)


def mode_action(env, t, mode_idx, phase0=0.0):
    """按第 mode_idx 套参数生成归一化动作（与 env.action_space 同量纲）。

    env  : ShadowHandReorientEnv 实例（要用它的 q_grasp 作工作点）
    t    : 物理时间（秒）
    phase0: 该 episode 的行波初相位——这就是观测里看不见的那个隐变量
    """
    amp, freq, wave, skew, thc, j1c = MODES[mode_idx][1:]
    w = 2.0 * math.pi * freq
    q = env.q_grasp.copy()
    for k, i0 in enumerate(H.IDX_FINGER_J4):
        th = w * t + wave * k + phase0
        s = math.sin(th) + skew * math.sin(2.0 * th)      # 非对称波形 -> 每周期留下净转动
        q[i0 + 2] += amp * s
        q[i0 + 3] += amp * j1c * s
    q[H.IDX_THJ1] += thc * amp * math.sin(w * t + phase0)
    return np.clip((q - env.q_grasp) / ACTION_SCALE, -1.0, 1.0).astype(np.float32)


def make_policy(env, mode_idx, phase0=0.0):
    """包成 (env, obs) -> action 的形式，便于塞进统一评测循环。"""
    from hand_env import CTRL_DT

    def pol(e, obs):
        return mode_action(e, e._t * CTRL_DT, mode_idx, phase0)
    return pol
