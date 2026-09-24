#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPC 与阻尼最小二乘 IK 的定量对比
================================

任务
----
同一只 Shadow Hand，五个指尖跟踪一条已知的连续目标轨迹（世界系，
振幅 2 cm、频率 0.8 Hz）。控制器每 20 ms 出一次结果，中间 10 个物理步。

两个控制器
----------
**DLS-IK（阻尼最小二乘，01/02/04 里用的就是这套）**
    纯反应式：只看当前误差，求解 dq = Jᵀ(JJᵀ+λ²I)⁻¹e。没有约束、没有预测。
    冗余度用阻尼项处理。它的固有缺点是**跟踪滞后**——误差为零时它就不动了，
    而目标一直在动，所以永远落后一步。

**MPC（casadi + IPOPT）**
    预测型：以当前构型处的雅可比做线性化（速度级模型 p_{k+1} = p_k + J·dq_k），
    在预测窗 N=10 内最小化跟踪误差 + 控制量，
    并且**显式带约束**：关节限位、单步控制量上限。

    它知道未来的参考轨迹（这是它能赢的根本原因，必须讲清楚；
    如果只给当前参考点，MPC 的优势就只剩约束处理了）。

    诚实标注：这里的模型是"冻结雅可比"的线性化模型，每控制步重新线性化一次，
    不是完整的非线性 MPC。这是工程上常用的做法，量级上足以体现预测+约束的价值。

代价
----
MPC 不是免费的：每步要在线解一个 NLP。本脚本把**求解耗时**和**跟踪误差**
同时报出来——这才是选型时真正要看的两条曲线（精度 vs 算力）。

用法
----
    python mpc_control.py --steps 150 --horizon 10
"""
from __future__ import annotations

import argparse
import json
import os
import time

import casadi as ca
import numpy as np
import mujoco

import hand_common as H
from plot_style import use_cjk, CLR_PRIMARY, CLR_ACC, CLR_WARN, CLR_EXPERT, CLR_GRAY
use_cjk()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CTRL_DT = 0.020          # 外环 50 Hz
PHYS_PER_CTRL = 10       # 每个控制步 10 个物理步 = 20 ms
DLS_MAX_DQ = 0.06        # 单步关节增量上限 (rad)：约 3 rad/s 关节速度，故意收紧到让"预测"有价值
MPC_W_POS = 1.0e4        # 位置误差权重 (m^-2)
MPC_W_DQ = 1.0e-2        # 控制量权重
AMP = 0.012              # 目标轨迹振幅 (m)
FREQ = 1.5               # 目标轨迹频率 (Hz)
# 起始姿态与约束都往关节行程内部收一点。踩过的坑：直接用 keyframe 姿态起步时，
# 不少关节已经贴在上限上，而硬约束 lo ≤ q0 + Σdq ≤ hi 对这些关节只剩单向自由度，
# 一旦参考轨迹要求反向运动就整个问题不可行，IPOPT 返回的不可行解会让 MPC
# 表现比纯反应式的 DLS-IK 还差 3 倍（这不是调参问题，是建模问题）。
POS_MARGIN = 0.10        # 起始姿态离关节上下限至少保留 10% 行程
LIM_MARGIN = 0.02        # 约束里的限位内缩 2%，给数值误差留余量


# ------------------------------------------------------------------ 模型
class HandRig:
    def __init__(self, key="open hand"):
        self.m = mujoco.MjModel.from_xml_path(H.SCENE)
        self.d = mujoco.MjData(self.m)
        self.ctrl = H.HandController(self.m, self.d)
        keys = H.parse_keyframes()
        self.q_key = np.asarray(keys[key], float)
        self.tips = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in H.TIP_BODIES]
        self.nu = len(self.ctrl.dof)
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        # 起始姿态收到关节行程内部，保证两个方向都有运动余量（见 POS_MARGIN 注释）
        span = self.ctrl.hi - self.ctrl.lo
        self.q_start = np.clip(self.q_key,
                               self.ctrl.lo + POS_MARGIN * span,
                               self.ctrl.hi - POS_MARGIN * span)
        self.lo_m = self.ctrl.lo + LIM_MARGIN * span
        self.hi_m = self.ctrl.hi - LIM_MARGIN * span
        self.reset()

    def reset(self):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.ctrl.qadr] = self.q_start
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, d)
        for _ in range(300):
            self.ctrl.drive(self.q_start)
            mujoco.mj_step(self.m, d)
        mujoco.mj_forward(self.m, d)

    def tip_pos(self):
        return np.concatenate([self.d.xpos[b] for b in self.tips])

    def jac(self):
        rows = []
        for b in self.tips:
            mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr,
                          self.d.xpos[b], b)
            rows.append(self.jacp[:, self.ctrl.dof])
        return np.vstack(rows)                            # (15, 24)

    def drive_to(self, q, n_phys=PHYS_PER_CTRL):
        q = np.clip(q, self.ctrl.lo, self.ctrl.hi)
        for _ in range(n_phys):
            self.ctrl.drive(q)
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        return q

    def q(self):
        return self.d.qpos[self.ctrl.qadr].copy()


def reference(t, p0):
    """目标轨迹：每个指尖在 x-z 平面画正弦，相邻指尖有固定相位差。"""
    out = p0.copy()
    for i in range(5):
        ph = 2 * np.pi * FREQ * t + 0.6 * i
        out[3 * i + 0] += AMP * np.sin(ph)
        out[3 * i + 2] += 0.5 * AMP * np.sin(ph)
    return out


# ------------------------------------------------------------------ DLS-IK
def dls_step(J, err, lam=1e-3):
    """一步阻尼最小二乘：无约束、无预测。"""
    A = J @ J.T + lam * np.eye(J.shape[0])
    dq = J.T @ np.linalg.solve(A, err)
    return np.clip(dq, -DLS_MAX_DQ, DLS_MAX_DQ)


# ------------------------------------------------------------------ MPC
class LinearMPC:
    """速度级线性 MPC。支持两种后端，用来对比"建模对了、求解器选错了"的代价。

    问题（每控制步在当前构型重新线性化一次）：

        p_{k+1} = p_k + J·dq_k ,   q_{k+1} = q_k + dq_k
        min  Σ_k  w_p·‖p_ref,k − p_k‖²  +  w_dq·‖dq_k‖²
        s.t. |dq_k| ≤ dq_max                       （执行器速度上限）
             lo ≤ q_0 + Σ_{j≤k} dq_j ≤ hi          （关节限位）

    **后端 1：通用 NLP 求解器（casadi + IPOPT）**
        直接把上面写成 NLP 丢进去。IPOPT 是通用非线性求解器，
        它不知道这是个凸 QP，也不知道问题的块稀疏结构。
        实测 600 ms/步——对一个 20 ms 的控制周期完全不可用。

    **后端 2：消元成标准 QP + 专用求解器（OSQP）**
        这个问题的代价是二次的、约束是线性的，**它就是 QP**。
        把决策变量 U = [dq_0; …; dq_{N−1}] 展开，预测位置可以写成

            P = 1_N⊗p_0 + G·U ,   G = (I_N ⊗ J)·L

        其中 L 是"块下三角全 1"的累积矩阵（把 dq 累加成 q_k）。
        于是代价整理成标准形式 ½UᵀHU + gᵀU + const：

            H = 2·(w_p·GᵀG + w_dq·I)
            g = −2·w_p·Gᵀ(P_ref − 1_N⊗p_0)

        约束同理拼成 [L; I]·U ∈ [l, u]。OSQP 就是为这种问题设计的
        （一阶 ADMM、支持热启动），实测把单步求解压到个位数毫秒。

    顺带一提：约束里关节限位是**关于 U 的线性约束**（q_k 累积了前面的 dq），
    不能写成变量界；踩过这个坑，写成变量界会让限位完全失效。
    """

    def __init__(self, N=8, nu=24, nt=15, max_dq=DLS_MAX_DQ, backend="osqp"):
        self.N, self.nu, self.nt, self.max_dq = N, nu, nt, max_dq
        self.backend = backend
        # 决策变量按"块"排列：U = [dq_0; dq_1; ...]，每块 nu 维
        self.L = np.kron(np.tril(np.ones((N, N))), np.eye(nu))     # (nu·N, nu·N)
        self.idn = np.eye(nu * N)
        if backend == "osqp":
            import osqp
            self.osqp = osqp
            self._prob = None
        else:
            self.build_casadi()

    # ---------------- 后端 1：IPOPT
    def build_casadi(self):
        N, nu, nt = self.N, self.nu, self.nt
        J = ca.MX.sym("J", nt, nu)
        q0 = ca.MX.sym("q0", nu)
        p0 = ca.MX.sym("p0", nt)
        Pref = ca.MX.sym("Pref", nt, N)
        lo = ca.MX.sym("lo", nu)
        hi = ca.MX.sym("hi", nu)
        DQ = ca.MX.sym("DQ", nu, N)
        cost = 0
        dq_cum = ca.MX.zeros(nu, 1)
        g = []
        for k in range(N):
            dq_cum = dq_cum + DQ[:, k]
            e = Pref[:, k] - (p0 + J @ dq_cum)
            cost += MPC_W_POS * ca.dot(e, e) + MPC_W_DQ * ca.dot(DQ[:, k], DQ[:, k])
            g.append(q0 + dq_cum)
        nlp = {"x": ca.reshape(DQ, nu * N, 1), "f": cost, "g": ca.vertcat(*g),
               "p": ca.vertcat(ca.reshape(J, nt * nu, 1), q0, p0,
                               ca.reshape(Pref, nt * N, 1), lo, hi)}
        self.nlpsol = ca.nlpsol(
            "mpc", "ipopt", nlp,
            {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
             "ipopt.max_iter": 60, "ipopt.tol": 1e-3,
             "ipopt.hessian_approximation": "limited-memory",
             "ipopt.warm_start_init_point": "yes"})
        self.lbx = [-self.max_dq] * (nu * N)
        self.ubx = [self.max_dq] * (nu * N)

    def _solve_ipopt(self, J, q0, p0, Pref, lo, hi, warm):
        N, nu = self.N, self.nu
        p = np.concatenate([J.ravel(order="F"), q0, p0,
                            Pref.ravel(order="F"), lo, hi])
        x0 = np.zeros(nu * N) if warm is None else warm
        r = self.nlpsol(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx,
                        lbg=np.tile(lo, N), ubg=np.tile(hi, N))
        return np.asarray(r["x"]).reshape(nu, N, order="F"), float(r["f"])

    # ---------------- 后端 2：OSQP
    def _solve_osqp(self, J, q0, p0, Pref, lo, hi, warm):
        from scipy import sparse
        N, nu, nt = self.N, self.nu, self.nt
        IJ = np.kron(np.eye(N), J)                     # (nt·N, nu·N)
        G = IJ @ self.L                                # (nt·N, nu·N)
        ref = Pref.reshape(-1, order="F")
        r = ref - np.tile(p0, N)
        H = 2.0 * (MPC_W_POS * (G.T @ G) + MPC_W_DQ * self.idn)
        gq = -2.0 * MPC_W_POS * (G.T @ r)
        A = np.vstack([self.L, self.idn])
        l = np.concatenate([np.tile(lo - q0, N), -np.full(nu * N, self.max_dq)])
        u = np.concatenate([np.tile(hi - q0, N), np.full(nu * N, self.max_dq)])
        P = sparse.csc_matrix(H)
        A = sparse.csc_matrix(A)
        if self._prob is None:
            self._prob = self.osqp.OSQP()
            self._prob.setup(P=P, q=gq, A=A, l=l, u=u, verbose=False,
                             eps_abs=1e-3, eps_rel=1e-3, max_iter=1000,
                             polish=False, warm_start=True)
        else:
            # 稀疏结构不变，只更新数值；比重新 setup 便宜
            self._prob.update(P=P.data, q=gq, l=l, u=u, Ax=A.data)
        res = self._prob.solve()
        U = np.asarray(res.x).reshape(N, nu)
        DQ = U.T.copy()                               # (nu, N)
        # 代价值（用于诊断，与 IPOPT 口径一致）
        f = 0.5 * U.ravel() @ H @ U.ravel() + gq @ U.ravel()
        return DQ, float(f)

    def solve(self, J, q0, p0, Pref, lo, hi, warm=None):
        if self.backend == "osqp":
            return self._solve_osqp(J, q0, p0, Pref, lo, hi, warm)
        return self._solve_ipopt(J, q0, p0, Pref, lo, hi, warm)


# ------------------------------------------------------------------ 实验
def run(kind, steps=150, horizon=8, backend="osqp"):
    rig = HandRig()
    p0 = rig.tip_pos()
    mpc = LinearMPC(N=horizon, max_dq=DLS_MAX_DQ, backend=backend) if kind == "mpc" else None
    q = rig.q()
    errs, times, limits = [], [], []
    warm = None
    for k in range(steps):
        t = k * CTRL_DT
        t0 = time.perf_counter()
        if kind == "dls":
            J = rig.jac()
            e = reference(t + CTRL_DT, p0) - rig.tip_pos()
            dq_seq = dls_step(J, e)
        else:
            J = rig.jac()
            # 预测窗内的参考轨迹（MPC 知道未来 —— 这是它领先的根本原因）
            Pref = np.stack([reference(t + (j + 1) * CTRL_DT, p0)
                             for j in range(horizon)], axis=1)
            DQ, _f = mpc.solve(J, q, rig.tip_pos(), Pref,
                               rig.lo_m, rig.hi_m, warm)
            warm = DQ.T.reshape(-1).copy()        # 热启动：块序 U = [dq_0; dq_1; …]
            dq_seq = DQ[:, 0]
        solve_ms = (time.perf_counter() - t0) * 1000
        q_new = rig.drive_to(q + dq_seq)
        span = rig.ctrl.hi - rig.ctrl.lo
        limits.append(int(np.sum((q_new <= rig.ctrl.lo + 0.01 * span) |
                                 (q_new >= rig.ctrl.hi - 0.01 * span))))
        q = q_new
        err = reference(t + CTRL_DT, p0) - rig.tip_pos()
        errs.append(np.linalg.norm(err.reshape(5, 3), axis=1) * 1000)  # 每指尖 mm
        times.append(solve_ms)
    errs = np.asarray(errs)                      # (T, 5)
    return dict(per_tip_rms_mm=errs.mean(0).tolist(),
                rms_mm=float(np.sqrt((errs ** 2).mean())),
                max_mm=float(errs.max()),
                solve_ms_mean=float(np.mean(times)),
                solve_ms_p95=float(np.percentile(times, 95)),
                limit_touch_steps=int(sum(1 for v in limits if v > 0)),
                err_series=errs.mean(1))


def main():
    ap = argparse.ArgumentParser(description="MPC vs DLS-IK 指尖轨迹跟踪")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--ipopt-steps", type=int, default=10,
                    help="IPOPT 后端只跑这么多步用于对比求解耗时")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    print("=" * 78)
    print(f"MPC vs DLS-IK · 五指指尖轨迹跟踪 · {args.steps} 个控制步"
          f"（{CTRL_DT * 1000:.0f} ms/步，目标 {FREQ} Hz / 振幅 {AMP * 1000:.1f} mm）"
          f"· MPC 预测窗 {args.horizon}")
    print("=" * 78)

    cases = [("dls", "dls", "DLS-IK（反应式，无预测、无约束）"),
             ("osqp", "mpc", "MPC（预测 + 约束）· 消元成 QP + OSQP"),
             ("ipopt", "mpc", "MPC（同一问题）· 通用 NLP 求解器 IPOPT")]
    out = {}
    for key, kind, label in cases:
        steps = args.ipopt_steps if key == "ipopt" else args.steps
        r = run(kind, steps=steps, horizon=args.horizon,
                backend="ipopt" if key == "ipopt" else "osqp")
        out[key] = r
        print(f"  {label}")
        print(f"    跟踪 RMS {r['rms_mm']:7.3f} mm | 最大 {r['max_mm']:7.3f} mm | "
              f"求解 {r['solve_ms_mean']:8.2f} ms/步 (p95 {r['solve_ms_p95']:7.2f}) | "
              f"触及限位步数 {r['limit_touch_steps']}/{steps}")

    imp = (out["dls"]["rms_mm"] - out["osqp"]["rms_mm"]) / out["dls"]["rms_mm"] * 100
    print(f"\n  => MPC(QP/OSQP) 相对 DLS-IK 的跟踪 RMS 变化：{-imp:+.1f}%")
    print(f"     代价：{out['osqp']['solve_ms_mean']:.2f} ms/步 在线求解"
          f"（DLS-IK {out['dls']['solve_ms_mean']:.3f} ms，"
          f"慢 {out['osqp']['solve_ms_mean'] / max(out['dls']['solve_ms_mean'], 1e-9):.0f} 倍）"
          f"，控制周期 {CTRL_DT * 1000:.0f} ms -> "
          f"p50 {'达标' if out['osqp']['solve_ms_mean'] < CTRL_DT * 1000 else '超标'}"
          f" / p95 {'达标' if out['osqp']['solve_ms_p95'] < CTRL_DT * 1000 else '超标'}"
          f"（硬实时看 p95，不能只看中位数）")
    print(f"     求解器选型：同一 MPC 问题用 IPOPT 要 "
          f"{out['ipopt']['solve_ms_mean']:.1f} ms/步，换成 QP 专用的 OSQP 只要 "
          f"{out['osqp']['solve_ms_mean']:.2f} ms/步，快 "
          f"{out['ipopt']['solve_ms_mean'] / max(out['osqp']['solve_ms_mean'], 1e-9):.0f} 倍。")

    # ---------------- 图
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    t = np.arange(args.steps) * CTRL_DT
    ax[0].plot(t, out["dls"]["err_series"], lw=1.7, color=CLR_EXPERT,
               label="DLS-IK")
    ax[0].plot(t, out["osqp"]["err_series"], lw=1.7, color=CLR_ACC,
               label="MPC (OSQP)")
    ax[0].set_title("五指平均跟踪误差")
    ax[0].set_xlabel("时间 (s)")
    ax[0].set_ylabel("mm")
    ax[0].legend(fontsize=9)

    xs = np.arange(5)
    w = 0.36
    ax[1].bar(xs - w / 2, out["dls"]["per_tip_rms_mm"], width=w, color=CLR_EXPERT,
              label="DLS-IK")
    ax[1].bar(xs + w / 2, out["osqp"]["per_tip_rms_mm"], width=w, color=CLR_ACC,
              label="MPC (OSQP)")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(["食", "中", "无名", "小", "拇指"])
    ax[1].set_title("各指尖跟踪 RMS")
    ax[1].set_ylabel("mm")
    ax[1].legend(fontsize=9)

    bars = ax[2].bar([0, 1, 2], [out["dls"]["rms_mm"], out["osqp"]["rms_mm"],
                                 out["osqp"]["solve_ms_mean"]],
                     color=[CLR_EXPERT, CLR_ACC, CLR_WARN], width=0.55)
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["DLS-IK\nRMS(mm)", "MPC\nRMS(mm)", "MPC\n求解(ms)"],
                          fontsize=9)
    for b, v in zip(bars, [out["dls"]["rms_mm"], out["osqp"]["rms_mm"],
                           out["osqp"]["solve_ms_mean"]]):
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                   ha="center", va="bottom", fontsize=9)
    ax[2].set_title("精度 vs 算力")
    ax[2].margins(y=0.2)

    names = ["DLS-IK", "IPOPT", "OSQP"]
    vals = [out["dls"]["solve_ms_mean"], out["ipopt"]["solve_ms_mean"],
            out["osqp"]["solve_ms_mean"]]
    bars = ax[3].bar(range(3), vals, color=[CLR_EXPERT, CLR_WARN, CLR_ACC],
                     width=0.55)
    ax[3].axhline(CTRL_DT * 1000, ls="--", lw=1.3, color=CLR_PRIMARY,
                  label=f"控制周期 {CTRL_DT * 1000:.0f} ms")
    ax[3].set_xticks(range(3))
    ax[3].set_xticklabels(names)
    ax[3].set_yscale("log")
    ax[3].set_ylabel("单步求解耗时 (ms, 对数轴)")
    ax[3].set_title("求解器选型决定可部署性")
    for b, v in zip(bars, vals):
        ax[3].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                   ha="center", va="bottom", fontsize=9)
    ax[3].legend(fontsize=8.5)
    ax[3].margins(y=0.28)

    fig.suptitle(f"Shadow Hand 指尖轨迹跟踪 · MPC（窗 {args.horizon}）与 DLS-IK 对比",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "mpc_vs_ik.png"))
    plt.close(fig)
    print("\n图 -> results/mpc_vs_ik.png")

    with open(os.path.join(RES, "mpc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(steps=args.steps, horizon=args.horizon, ctrl_dt=CTRL_DT,
                       freq_hz=FREQ, amp_m=AMP, max_dq=DLS_MAX_DQ,
                       rms_change_pct=-imp, ipopt_steps=args.ipopt_steps,
                       results={k: {kk: vv for kk, vv in v.items()
                                    if kk != "err_series"}
                                for k, v in out.items()}),
                  f, ensure_ascii=False, indent=2)
    print("指标 -> results/mpc_metrics.json")


if __name__ == "__main__":
    main()
