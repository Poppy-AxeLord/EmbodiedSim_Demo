#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
刚体动力学：手写实现与引擎对拍
==============================

为什么要做这个
--------------
前面几个项目里，动力学部分都是「**用**引擎」的姿态：调 `qfrc_bias` 拿重力/科氏力，
调 `mj_step` 推进物理。这足以写出正确的计算力矩控制器与 MPC，但严格说那是
「应用动力学」，不是「实现动力学算法」——两者在追问下会被区分开。

本脚本把那一层补上：**用定义式手写质量矩阵与逆动力学，再与引擎里两个独立算法
逐元素数值对拍**：

    mj_fullM  ←→  CRBA（Composite Rigid Body Algorithm，复合刚体算法）
    mj_rne    ←→  RNEA（Recursive Newton-Euler，递归牛顿-欧拉）

对拍的意义在于：手写实现与引擎实现在**算法路径上完全不同**（一个是全矩阵组装，
一个是递归递推），如果两者在机器精度上吻合，就同时验证了两边。

自研的三件事，以及边界
----------------------
**1. 质量矩阵 M(q)** —— 真正自研，用定义式：

        M(q) = Σ_i [ m_i · J_pᵢᵀ J_pᵢ  +  J_rᵢᵀ · I_i(world) · J_rᵢ ]

    对每个 body 累加「质心线速度雅可比」与「角速度雅可比」的二次型。
    这就是质量矩阵的教科书定义（系统动能 = ½ q̇ᵀMq̇ 展开后的二次型），
    写出来只需要雅可比（引擎给）和 body 惯量参数（引擎给），
    不依赖任何动力学算法。

**2. 逆动力学** τ = M(q)·q̈ + C(q,q̇)·q̇ + g(q)
    其中 M 用上面手写的那个；偏置项取 `qfrc_bias`。
    **诚实边界**：偏置项没有从零手写。从零算 C·q̇ + g 需要完整实现 RNEA
    的空间向量递推，代码量数倍于此处，而对拍结论不会更强（它本身就是要
    被验证的对象）。所以这里明确标注：**M 是自研的，偏置项是引擎的**。

**3. 偏置力分解** —— 验证 qfrc_bias(q̇) = qfrc_bias(0) + C(q,q̇)·q̇
    把偏置项拆成「重力项」与「科氏/离心项」，这一条用来确认我们对 qfrc_bias
    里到底装了什么的理解是对的（它不含关节阻尼、不含摩擦力）。

两条由对拍逼出来的工程结论
--------------------------
**① `mj_rne(flg_acc=0)` 逐位等于 `qfrc_bias`**（实测偏差 0.000e+00）。
    这是判断「RNEA 调用链是否正确」的哨兵——不成立的话，后面所有对拍都无意义。

**② `mj_fullM`(CRBA) 含转子惯量 armature，`mj_rne`(RNEA) 不含。**
    第一版手写 M 没加 armature，对拍给出 3.3e-3 的相对偏差；再查发现差异
    **全部落在对角线上**、且与 `diag(dof_armature)` 逐位吻合；补上后降到机器精度。
    但补上之后，逆动力学的对拍误差反而从 1e-16 涨到 1.5e-2——因为 RNEA
    本来就不含 armature。**两条约定都对，只是不一样。**
    所以正确做法是：对拍 CRBA 用含 armature 的 M，对拍 RNEA 用不含的；
    两者之差恰好是 `armature ⊙ q̈`，脚本里把这一条也单独量了一次。

    这是个真实的坑：如果只写实现、不做对拍，这 2e-4 会一直藏着，
    直到某天用它算逆动力学时，表现为一个说不清来源的力矩偏差。

用法
----
    python rigid_body_dynamics.py --states 24
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import mujoco

import hand_common as H
from hand_env import ShadowHandReorientEnv
from plot_style import (use_cjk, CLR_PRIMARY, CLR_EXPERT, CLR_ACC, CLR_WARN,
                        CLR_GRAY)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


# ------------------------------------------------------------------ 自研实现
def body_inertia_world(model, data, bid):
    """返回 body 的质量与**质心处、世界朝向**的 3x3 惯量矩阵。

    MuJoCo 里 body 的惯量是存在主轴系下的对角阵（`body_inertia`），
    主轴系到世界的旋转由 `data.ximat` 给出，所以

        I_world = R · diag(I_principal) · Rᵀ

    这是把对角惯量搬到世界系的唯一正确做法；直接拿 body_inertia 当世界系惯量
    是常见错误（只有 body 主轴恰好与世界对齐时才凑巧成立）。
    """
    mass = float(model.body_mass[bid])
    R = data.ximat[bid].reshape(3, 3)
    I_loc = np.diag(np.asarray(model.body_inertia[bid], float))
    return mass, R @ I_loc @ R.T


def mass_matrix_manual(model, data, include_armature=True):
    """自研质量矩阵：M = Σ_i ( m_i·J_pᵀJ_p + J_rᵀ·I_i·J_r ) [+ diag(armature)]。

    雅可比用引擎的 `mj_jacBodyCom`（**质心**版本，不是 body 原点版本——
    惯量是关于质心的，用 `mj_jacBody` 会引入一个平移项而算错）。
    引擎只提供「材料」（雅可比 + 惯量参数），组装这个矩阵是这里做的事。

    **关于 armature（这一步是对拍才逼出来的）**：
    MuJoCo 的质量矩阵在惯量积分之外，还会把 `model.dof_armature` 加到对角线
    上（模拟电机转子惯量，本模型设成 2e-4）。第一版没加，对拍直接给出 3.3e-3
    的相对偏差；再一查，差异**全部落在对角线上**，且与 `diag(dof_armature)`
    逐位吻合。补上之后误差降到机器精度。

    这正是对拍的价值所在：它把「引擎与手写在约定上的差异」这类看不见的东西，
    变成了一个可归因的数字。如果只写实现、不做对拍，这个 2e-4 会一直藏着，
    直到某天用它算逆动力学时表现为一个莫名其妙的力矩偏差。
    """
    nv = model.nv
    M = np.zeros((nv, nv))
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    for bid in range(1, model.nbody):          # 0 是 world，无质量
        mass, I_w = body_inertia_world(model, data, bid)
        if mass <= 0.0:                        # 纯参考系 body（无质量）
            continue
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, bid)
        M += mass * (jacp.T @ jacp) + (jacr.T @ I_w @ jacr)
    if include_armature:
        M += np.diag(np.asarray(model.dof_armature, float))
    return M


def engine_mass_matrix(model, data):
    """引擎质量矩阵（CRBA，结果缓存在 data.M 的稀疏三角形式里，这里展开成稠密）。

    注意签名差异：MuJoCo 3.x 是 `mj_fullM(m, d, dst)`，
    旧版本是 `mj_fullM(m, dst, d.qM)` —— 第二个参数从稀疏矩阵换成了 MjData。
    """
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return M


def rel_err(a, b):
    """全局相对误差：按参考量的最大幅值归一。

    不用逐元素相对误差——质量矩阵与力矩向量里有大量接近 0 的元素，
    逐元素相对误差会被 0 附近的小量放大成毫无意义的数字。
    """
    den = max(float(np.abs(b).max()), 1e-12)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / den)


# ------------------------------------------------------------------ 状态采样
def collect_states(env, n, rng):
    """采一批物理可达的状态。

    每个状态都由 `env.reset(seed=...)` 真实抓握后得到（球落在掌窝里、
    手指贴住球），再叠加随机关节偏移与随机速度，让状态分布覆盖得更宽，
    而不是只在一个构型附近打转。
    """
    states = []
    qadr = env.ctrl.qadr          # 24 个手部关节的 qpos 地址
    dadr = env.ctrl.dof           # 对应的 dof 地址
    for k in range(n):
        env.reset(seed=1000 + k)
        d = env.data
        qpos = d.qpos.copy()
        qvel = d.qvel.copy()
        # 关节角小幅扰动（含往限位方向推的样本，考验 M 在边界附近的表现）
        qpos[qadr] = np.clip(
            qpos[qadr] + rng.normal(0.0, 0.05, qadr.size), env.q_lo, env.q_hi)
        qvel[dadr] += rng.normal(0.0, 1.0, dadr.size)
        qvel[:6] += rng.normal(0.0, 0.05, 6)      # 球的速度也动一下
        states.append((qpos, qvel))
    return states


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="刚体动力学手写实现与引擎对拍")
    ap.add_argument("--states", type=int, default=24, help="对拍的状态数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acc-scale", type=float, default=20.0,
                    help="随机广义加速度的尺度 (rad/s²)")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = ShadowHandReorientEnv(randomize=True)
    m, d = env.model, env.data
    nv, nbody = m.nv, m.nbody
    n_hand = env.n_hand

    print("=" * 70)
    print(f"刚体动力学对拍 · {args.states} 个状态 · nv={nv} nbody={nbody}")
    print("=" * 70)
    print(f"手部关节 {n_hand} 个 + 球自由关节 6 个 = {nv} 个自由度")
    print()

    # ---- 哨兵：先确认 mj_rne 的调用链是对的 ----------------------------
    env.reset(seed=0)
    mujoco.mj_forward(m, d)
    tau_rne0 = np.zeros(nv)
    mujoco.mj_rne(m, d, 0, tau_rne0)          # flg_acc=0 ⇒ 只算偏置项
    sentinel = float(np.abs(tau_rne0 - d.qfrc_bias).max())
    print(f"[哨兵] mj_rne(flg_acc=0) 与 qfrc_bias 的最大偏差：{sentinel:.3e}")
    print("        （逐位相等才说明 RNEA 调用链正确，后面所有对拍才有意义）")
    print()

    states = collect_states(env, args.states, rng)

    rec = {
        "mass_max_abs": [], "mass_rel": [], "mass_sym": [],
        "eig_min": [], "eig_max": [],
        "tau_rel": [], "tau_abs": [],
        "coriolis_quad": [], "arm_slope": [],
        "time_manual_ms": [], "time_engine_ms": [],
    }
    bias_gravity = bias_coriolis = None
    M_last = M_engine_last = None

    for i, (qpos, qvel) in enumerate(states):
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)                # 填充 qM / cdof / cinert / qfrc_bias

        # 随机广义加速度（量级对齐真实控制场景：几十 rad/s²）
        qacc = rng.normal(0.0, args.acc_scale, nv)

        # --- 1) 质量矩阵：自研 vs 引擎 CRBA ---
        t0 = time.perf_counter()
        M_man = mass_matrix_manual(m, d)
        t1 = time.perf_counter()
        M_eng = engine_mass_matrix(m, d)
        t2 = time.perf_counter()

        # 一次性归因：不含转子惯量时差多少、差异是否只在对角线、能否被 armature 解释
        if i == 0:
            M_bare = mass_matrix_manual(m, d, include_armature=False)
            arm_diag = np.diag(np.asarray(m.dof_armature, float))
            delta = M_eng - M_bare
            arm_info = {
                "armature_value": float(np.unique(m.dof_armature)[-1]),
                "n_dof_with_armature": int((m.dof_armature > 0).sum()),
                "offset_max": float(np.abs(delta).max()),
                "offdiag_max": float(
                    np.abs(delta - np.diag(np.diag(delta))).max()),
                "armature_attrib_resid": float(np.abs(delta - arm_diag).max()),
            }

        rec["mass_max_abs"].append(float(np.abs(M_man - M_eng).max()))
        rec["mass_rel"].append(rel_err(M_man, M_eng))
        rec["mass_sym"].append(float(np.abs(M_man - M_man.T).max()))
        ev = np.linalg.eigvalsh(0.5 * (M_man + M_man.T))
        rec["eig_min"].append(float(ev.min()))
        rec["eig_max"].append(float(ev.max()))
        rec["time_manual_ms"].append((t1 - t0) * 1e3)
        rec["time_engine_ms"].append((t2 - t1) * 1e3)

        # --- 2) 逆动力学：τ = M·q̈ + C·q̇ + g ---
        #     引擎内部有个不一致必须知道：mj_fullM(CRBA) **含** armature，
        #     而 mj_rne(RNEA) **不含**。所以要和 RNEA 对拍，就得用不含 armature 的 M；
        #     两者之差本身也单独量一次（见【2b】），它是 armature ⊙ q̈。
        arm_vec = np.asarray(m.dof_armature, float)
        M_bare = mass_matrix_manual(m, d, include_armature=False)

        tau_rne = np.zeros(nv)
        d.qacc[:] = qacc
        mujoco.mj_rne(m, d, 1, tau_rne)            # 引擎 RNEA：τ = M q̈ + C q̇ + g

        tau_bare = M_bare @ qacc + d.qfrc_bias     # 自研 M（与 RNEA 同约定）+ 引擎偏置项
        tau_full = M_man @ qacc + d.qfrc_bias      # 自研 M（与 CRBA 同约定）+ 引擎偏置项

        rec["tau_rel"].append(rel_err(tau_bare, tau_rne))
        rec["tau_abs"].append(float(np.abs(tau_bare - tau_rne).max()))
        # 【2b】两个约定之差应当恰好等于 armature ⊙ q̈ —— 把引擎内部的不一致量化
        rec["arm_slope"].append(rel_err(tau_full - tau_bare, arm_vec * qacc))

        # --- 3) 偏置项分解，并验证科氏项的二次齐次性 ---
        #     qfrc_bias(q̇) = g(q) + C(q,q̇)·q̇
        #     把 q̇ 整体缩放 α 倍，科氏/离心项应按 α² 缩放（离心力 ∝ 速度平方）。
        #     这是**可独立证伪的物理性质**，而不是定义上的恒等式——
        #     如果 qfrc_bias 里混进了别的速度项（阻尼、摩擦），这里会立刻暴露。
        keep_v = d.qvel.copy()
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        bias_g = d.qfrc_bias.copy()                       # 纯重力项 g(q)
        d.qvel[:] = keep_v
        mujoco.mj_forward(m, d)
        bias_full = d.qfrc_bias.copy()
        cor1 = bias_full - bias_g                         # C(q,q̇)·q̇
        ALPHA = 2.5
        d.qvel[:] = ALPHA * keep_v
        mujoco.mj_forward(m, d)
        cor_a = d.qfrc_bias - bias_g
        rec["coriolis_quad"].append(rel_err(cor_a, ALPHA ** 2 * cor1))
        d.qvel[:] = keep_v
        mujoco.mj_forward(m, d)

        if i == len(states) // 2:
            bias_gravity = bias_g.copy()
            bias_coriolis = (bias_full - bias_g).copy()
        if i == 0:
            M_last, M_engine_last = M_man, M_eng
            qacc_first = qacc.copy()

    # ---- 汇总 -----------------------------------------------------------
    S = lambda k: np.asarray(rec[k], float)
    summary = {
        "mass_max_abs_err": float(S("mass_max_abs").max()),
        "mass_max_rel_err": float(S("mass_rel").max()),
        "mass_symmetry_err": float(S("mass_sym").max()),
        "eig_min": float(S("eig_min").min()),
        "eig_max": float(S("eig_max").max()),
        "tau_max_rel_err": float(S("tau_rel").max()),
        "tau_max_abs_err": float(S("tau_abs").max()),
        "armature_slope_rel_err": float(S("arm_slope").max()),
        "coriolis_quad_max_rel_err": float(S("coriolis_quad").max()),
        "sentinel_rne_vs_bias": sentinel,
        "manual_ms_mean": float(S("time_manual_ms").mean()),
        "engine_ms_mean": float(S("time_engine_ms").mean()),
    }

    print("【0】归因：手写实现与引擎的差异到底来自哪里？")
    print(f"      不带转子惯量时与引擎的偏差 {arm_info['offset_max']:.3e}"
          f"，其中非对角部分仅 {arm_info['offdiag_max']:.3e}"
          f"  ← 差异全在对角线上")
    print(f"      再减去 diag(dof_armature={arm_info['armature_value']:g}"
          f" × {arm_info['n_dof_with_armature']} 个自由度) 后残差 "
          f"{arm_info['armature_attrib_resid']:.3e}  ⇒ 差异就是转子惯量")
    print()
    print("【1】质量矩阵 自研 M = Σ(m·JpᵀJp + Jrᵀ·I·Jr) + diag(armature)"
          "  vs  引擎 mj_fullM(CRBA)")
    print(f"      最大绝对偏差 {summary['mass_max_abs_err']:.3e}"
          f"   最大相对偏差 {summary['mass_max_rel_err']:.3e}")
    print(f"      对称性偏差 ‖M−Mᵀ‖ = {summary['mass_symmetry_err']:.3e}")
    print(f"      特征值范围 [{summary['eig_min']:.3e}, {summary['eig_max']:.3e}]"
          f"   条件数 {summary['eig_max']/summary['eig_min']:.1f}")
    print()
    print("【2】逆动力学  τ = M·q̈ + C·q̇ + g   自研 M  vs  引擎 mj_rne(RNEA)")
    print(f"      最大绝对偏差 {summary['tau_max_abs_err']:.3e} N·m"
          f"   最大相对偏差 {summary['tau_max_rel_err']:.3e}")
    print()
    print("【2b】引擎内部约定检查：CRBA 与 RNEA 对 armature 的处理并不一致")
    print(f"      (M含armature·q̈) − (M不含armature·q̈) 与 armature⊙q̈ 的相对偏差 "
          f"{summary['armature_slope_rel_err']:.3e}")
    print("      ⇒ mj_fullM 含转子惯量、mj_rne 不含；写自己的动力学时必须知道这条")
    print()
    print("【3】偏置项分解  qfrc_bias(q̇) = g(q) + C(q,q̇)·q̇")
    print(f"      科氏项二次齐次性 C(αq̇)=α²C(q̇)，α=2.5 时残差 "
          f"{summary['coriolis_quad_max_rel_err']:.3e}")
    print()
    print(f"耗时：自研 M {summary['manual_ms_mean']:.2f} ms/次"
          f"   引擎 mj_fullM {summary['engine_ms_mean']:.2f} ms/次"
          f"   （{summary['manual_ms_mean']/summary['engine_ms_mean']:.1f}×）")

    out = {
        "config": {"states": args.states, "seed": args.seed, "nv": nv,
                   "nbody": nbody, "n_hand_joints": n_hand},
        "summary": summary,
        "armature_attribution": arm_info,
        "per_state": {k: [float(x) for x in v] for k, v in rec.items()},
    }
    with open(os.path.join(RES, "dynamics_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    plot(M_last, M_engine_last, rec, summary, bias_gravity, bias_coriolis,
         n_hand, nv)
    print()
    print(f"产出：{os.path.join(RES, 'dynamics_check.png')}")
    print(f"产出：{os.path.join(RES, 'dynamics_metrics.json')}")


# ------------------------------------------------------------------ 绘图
def plot(M_man, M_eng, rec, summary, bias_g, bias_c, n_hand, nv):
    use_cjk()
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.4))

    # (1) 质量矩阵热图（自研）+ 与引擎的差异色标
    a = ax[0, 0]
    im = a.imshow(M_man, cmap="Blues", aspect="auto")
    a.set_title(f"自研质量矩阵 M（{nv}×{nv}）\n"
                f"与引擎 CRBA 最大偏差 {summary['mass_max_abs_err']:.2e}",
                fontsize=11)
    a.set_xlabel("自由度 j")
    a.set_ylabel("自由度 i")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046)
    k = nv - 6
    a.axhline(k - 0.5, color=CLR_EXPERT, lw=0.8, ls="--")
    a.axvline(k - 0.5, color=CLR_EXPERT, lw=0.8, ls="--")
    a.text(nv * 0.72, nv * 0.20, "球\n(自由关节)", fontsize=9,
           color=CLR_EXPERT, ha="center", va="center")

    # (2) 逐状态逆动力学相对误差
    a = ax[0, 1]
    xs = np.arange(len(rec["tau_rel"])) + 1
    a.semilogy(xs, np.maximum(rec["tau_rel"], 1e-18), "o-",
               color=CLR_PRIMARY, ms=4.5, lw=1.4,
               label="自研 M  vs  mj_rne")
    a.semilogy(xs, np.maximum(rec["arm_slope"], 1e-18), "s--",
               color=CLR_ACC, ms=3.6, lw=1.1, alpha=0.85,
               label=r"(CRBA − RNEA) 的差值  vs  armature $\odot\,\ddot{q}$")
    a.axhline(1e-12, color=CLR_GRAY, lw=1.0, ls=":", label="float64 机器精度量级")
    a.set_xlabel("状态编号")
    a.set_ylabel("相对误差（按 ‖τ‖∞ 归一）")
    a.set_title(r"逆动力学 $\tau = M\ddot{q} + C\dot{q} + g$ 的逐状态吻合度",
                fontsize=11)
    a.legend(fontsize=8.5)
    a.margins(x=0.02)

    # (3) 特征值谱：验证 M 对称正定
    a = ax[1, 0]
    ev = np.linalg.eigvalsh(0.5 * (M_man + M_man.T))
    ev = np.sort(ev)[::-1]
    a.semilogy(np.arange(1, nv + 1), ev, "o-", color=CLR_PRIMARY, ms=4.0, lw=1.1)
    a.axhline(float(ev.max()), color=CLR_GRAY, lw=0.7, ls=":")
    a.set_xlabel("特征值序号（从大到小）")
    a.set_ylabel("特征值（对数轴）")
    a.set_title(r"质量矩阵对称正定：$\lambda_{min}$ 全程为正，无零/负特征值",
                fontsize=11)
    a.text(0.035, 0.10,
           f"$\\lambda_{{max}}$ = {ev.max():.3e}\n"
           f"$\\lambda_{{min}}$ = {ev.min():.3e}\n"
           f"条件数 {ev.max()/ev.min():.0f}",
           transform=a.transAxes, fontsize=9, va="bottom",
           bbox=dict(fc="white", ec=CLR_GRAY, alpha=0.85, lw=0.6))
    a.margins(x=0.03, y=0.18)

    # (4) 偏置项分解：重力 vs 科氏/离心
    a = ax[1, 1]
    idx = np.arange(nv)
    a.bar(idx - 0.2, bias_g, width=0.4, color=CLR_PRIMARY, label="重力项 $g(q)$")
    a.bar(idx + 0.2, bias_c, width=0.4, color=CLR_WARN,
          label=r"科氏/离心项 $C\dot{q}$")
    a.axvline(n_hand - 0.5, color=CLR_EXPERT, lw=0.9, ls="--")
    a.set_xlabel("自由度编号（左：24 个手部关节　右：球的 6 个）")
    a.set_ylabel("广义力 (N 或 N·m)")
    a.set_title("qfrc_bias 的分解：它只含重力与科氏力，不含阻尼与摩擦", fontsize=11)
    a.legend(fontsize=9)
    a.margins(x=0.02)

    fig.suptitle("手写刚体动力学与 MuJoCo 引擎对拍：两条独立算法路径的数值一致",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "dynamics_check.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
