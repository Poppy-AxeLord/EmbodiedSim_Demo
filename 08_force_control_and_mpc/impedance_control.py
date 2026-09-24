#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阻抗控制与力控：接触力峰值到底由什么决定
========================================

03 工程里那套"24 关节 PD 力矩 + 重力补偿"是**高刚度位置控制**：刚度 kp=2.5，
关节角被死死钉在期望值上。它在自由空间里很好用，但一碰到接触就出问题——
**高刚度 = 高接触力**。球被抓得越"死"，受到扰动时接触力冲击越大。

本脚本把两件事量化出来，而不是停留在"阻抗控制更柔顺"这种话术上：

1. **刚度扫描 + 扰动响应**
   维持握持姿态不动，给球加一个已知的阶跃外力，扫三档关节刚度：

     | 档位 | kp | kd | 期望 |
     |------|----|----|------|
     | 高刚度(03 默认) | 2.50 | 0.05 | 位移小、力峰大 |
     | 中阻抗         | 1.20 | 0.10 | 折中 |
     | 低刚度柔顺     | 0.50 | 0.15 | 力峰小、位移大 |

   指标：球位移峰值、接触力峰值、是否掉球、扰动后的回位时间。

2. **握持力闭环**
   "力控"这个词真正指的是：设定一个**期望接触力**，让控制器去实现它。
   这里用"抓握收紧程度" s ∈[0,1] 当控制量（q_des = q_grasp + s·(q_close − q_grasp)），
   先标定 s → 稳态接触力 的曲线，再用一个积分外环把接触力跟踪到阶跃设定值。

   （3D 里 ball 是均匀球体、接触点随姿态变化，所以接触力并不严格是 s 的
   单调线性函数；标定曲线会把这件事如实画出来。）

用法
----
    python impedance_control.py
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import mujoco

import hand_common as H
from plot_style import (use_cjk, CLR_PRIMARY, CLR_ACC, CLR_WARN, CLR_EXPERT,
                        CLR_GRAY, CLR_SEQ)
use_cjk()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CTRL_DT = 0.002
SQUEEZE_RAD = 0.20        # 单位挤压动作对应的关节增量 (rad)

STIFFNESS = [
    ("高刚度（03 默认）", 2.50, 0.05, CLR_EXPERT),
    ("中阻抗",           1.20, 0.10, CLR_WARN),
    ("低刚度柔顺",        0.50, 0.15, CLR_ACC),
]


def squeeze_direction():
    """单位挤压方向：四指 J2 屈曲 +1，拇指 J1 内收 -0.6。

    与 06/07 用的是**同一个动作参数化**（q_des = q_grasp + a·scale），
    只是这里 a 只沿一个固定方向，用标量 s 缩放。

    为什么不用 keyframe 里现成的 "grasp hard" 整体收紧：
    实测那样做会把球**挤出掌窝**——手指一路推向另一个姿态，球在中间被挤出去，
    接触构型退化，接触力反而从 10.9 N 掉到 9.5 N（还饱和在 s=1）。
    小幅、定向、围绕抓握工作点的挤压才是可用的握力控制量。
    """
    d = np.zeros(24)
    for i0 in H.IDX_FINGER_J4:
        d[i0 + 2] = 1.0
    d[H.IDX_THJ1] = -0.6
    return d / np.abs(d).max()


SQ_DIR = squeeze_direction()


# ------------------------------------------------------------------ 环境
class GraspRig:
    """握持球的最小测试台：加载模型、摆到抓握姿态、暴露接触力读数。"""

    def __init__(self, kp=2.5, kd=0.05):
        self.m = mujoco.MjModel.from_xml_path(H.SCENE)
        self.d = mujoco.MjData(self.m)
        self.ctrl = H.HandController(self.m, self.d, kp_finger=kp, kd_finger=kd)
        self.keys = H.parse_keyframes()
        self.q_grasp = np.asarray(self.keys["grasp sphere"], float)
        # "grasp hard" 是更紧的握持姿态，用作捏紧方向的目标；没有就退回 grasp sphere
        self.q_close = np.asarray(self.keys.get("grasp hard", self.keys["grasp sphere"]),
                                  float)
        self.ball = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                           "ball_geom")
        self.palm = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "lh_palm")
        # 球的自由关节在 qpos 里的真实地址（不能想当然按 0:3 取）
        bj = self.m.body_jntadr[self.ball]
        self.ball_qadr = int(self.m.jnt_qposadr[bj])
        if np.abs(self.q_close - self.q_grasp).max() < 1e-6:
            print("[warn] 找不到更紧的握持姿态，力标定将没有区分度")

    def set_stiffness(self, kp, kd):
        self.ctrl.kp[:] = kp
        self.ctrl.kd[:] = kd
        self.ctrl.kp[H.IDX_WRJ2] = self.ctrl.kp[H.IDX_WRJ1] = kp * 4.0
        self.ctrl.kd[H.IDX_WRJ2] = self.ctrl.kd[H.IDX_WRJ1] = kd * 8.0

    def reset(self, q_des=None, settle=400):
        d, m = self.d, self.m
        mujoco.mj_resetData(m, d)
        q = self.q_grasp.copy() if q_des is None else np.asarray(q_des, float)
        d.qpos[self.ctrl.qadr] = np.clip(q, self.ctrl.lo, self.ctrl.hi)
        d.qpos[self.ball_qadr:self.ball_qadr + 3] = [0.3330, 0.0011, 0.0094]
        d.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(settle):
            self.ctrl.drive(q)
            mujoco.mj_step(m, d)
        self.q_des = q
        return d.xpos[self.ball].copy()

    def contact_force(self):
        """球受到的法向接触力之和（N）。"""
        tot = 0.0
        buf = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.ball_geom in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                tot += abs(buf[0])
        return tot

    def step(self, q_des, dist_frc=(0, 0, 0)):
        self.d.xfrc_applied[self.ball, :3] = dist_frc
        self.ctrl.drive(q_des)
        mujoco.mj_step(self.m, self.d)
        return self.contact_force(), self.d.xpos[self.ball].copy()


# ------------------------------------------------------------------ 实验 1
def disturbance_response(kp, kd, force_n=6.0, t_on=0.10, t_off=0.30, steps=500):
    rig = GraspRig(kp, kd)
    rig.set_stiffness(kp, kd)
    p0 = rig.reset()
    # 扰动前的稳态接触力：低刚度下连"基线握力"都会更低，不扣掉基线就没法比
    base = np.mean([rig.step(rig.q_des)[0] for _ in range(50)])
    fx, fy = force_n, 0.0
    frc, disp = [], []
    for k in range(steps):
        t = k * CTRL_DT
        f = (fx, fy, 0.0) if t_on <= t < t_off else (0.0, 0.0, 0.0)
        c, p = rig.step(rig.q_des, f)
        frc.append(c)
        disp.append(float(np.linalg.norm(p - p0)))
    frc = np.asarray(frc)
    disp = np.asarray(disp)
    # 回位判据用相对量：扰动撤除后位移回到峰值的 5% 以内
    # （球受扰动会滚动、落进新的接触构型，要求回到 <0.5 mm 是不现实的）
    i_off = int(t_off / CTRL_DT)
    thr = max(0.0005, 0.05 * disp.max())
    rec = None
    for i, v in enumerate(disp[i_off:]):
        if v < thr:
            rec = i * CTRL_DT
            break
    return dict(base_force=float(base),
                peak_force=float(frc.max()),
                force_rise=float(frc.max() - base),
                steady_force=float(frc[-50:].mean()),
                peak_disp_mm=float(disp.max() * 1000),
                residual_disp_mm=float(disp[-50:].mean() * 1000),
                recovery_s=rec, dropped=bool(disp.max() > 0.16),
                force_series=frc, disp_series=disp)


# ------------------------------------------------------------------ 实验 2
def force_calibration(s_values, settle=300):
    """静态标定：挤压量 s → 稳态接触力。"""
    rig = GraspRig(2.5, 0.05)
    out = []
    for s in s_values:
        rig.set_stiffness(2.5, 0.05)
        q = rig.q_grasp + s * SQUEEZE_RAD * SQ_DIR
        rig.reset(q_des=q, settle=settle)
        for _ in range(100):
            rig.step(q)
        out.append(rig.contact_force())
    return np.asarray(out)


def force_tracking(setpoints, k_i=1.2, steps_per_sp=900, s_lo=0.0, s_hi=1.0,
                   verbose=True):
    """积分外环调挤压量 s，把接触力跟踪到设定值。

    **每个设定值都从复位后的静置握持开始**：让 s 连续挤压会把球逐渐挤离掌窝，
    接触构型退化，闭环看起来"失效"——那是滑移/挤出，不该混进"力跟踪精度"。

    控制量 s 的物理含义：q_des = q_grasp + s·SQUEEZE_RAD·SQ_DIR，
    接触力的可达区间由 force_calibration() 先标定出来，
    **设定值必须落在区间内**，否则控制器只会饱和。
    """
    out = []
    for sp in setpoints:
        rig = GraspRig(2.5, 0.05)
        rig.set_stiffness(2.5, 0.05)
        rig.reset()
        s = 0.0
        tt, ff, ss, sh = [], [], [], []
        for k in range(steps_per_sp):
            q = rig.q_grasp + s * SQUEEZE_RAD * SQ_DIR
            c, _p = rig.step(q)
            s = float(np.clip(s + k_i * (sp - c) * CTRL_DT, s_lo, s_hi))
            tt.append(k * CTRL_DT)
            ff.append(c)
            ss.append(sp)
            sh.append(s)
        tail = np.asarray(ff)[-100:]
        sat = sh[-1] in (s_lo, s_hi)
        out.append(dict(setpoint=sp, t=np.asarray(tt), force=np.asarray(ff),
                        sp=np.asarray(ss), s=np.asarray(sh),
                        steady=float(tail.mean()),
                        error=float(tail.mean() - sp), saturated=bool(sat)))
        if verbose:
            print(f"  设定 {sp:5.1f} N -> 稳态 {tail.mean():6.2f} N "
                  f"（误差 {tail.mean() - sp:+5.2f} N，"
                  f"{(tail.mean() - sp) / sp * 100:+5.1f}%）| 末态 s={sh[-1]:.3f}"
                  + ("  [饱和]" if sat else ""))
    return out


# ------------------------------------------------------------------ 绘图
def main():
    ap = argparse.ArgumentParser(description="阻抗控制与力控定量对比")
    ap.add_argument("--force", type=float, default=6.0, help="扰动力大小 (N)")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    print("=" * 66)
    print(f"阻抗控制 · 扰动力 {args.force:.1f} N")
    print("=" * 66)

    rows = {}
    for name, kp, kd, _c in STIFFNESS:
        r = disturbance_response(kp, kd, force_n=args.force)
        rows[name] = r
        print(f"  {name:16s} kp={kp:.2f} kd={kd:.2f} | "
              f"基线握力 {r['base_force']:5.2f} N | 力峰 {r['peak_force']:6.2f} N "
              f"(+{r['force_rise']:5.2f}) | 位移峰 {r['peak_disp_mm']:5.2f} mm | "
              f"残余位移 {r['residual_disp_mm']:5.2f} mm | "
              f"掉球 {'是' if r['dropped'] else '否'}")

    # ---------------- 力标定 + 力闭环
    print("\n握持力标定（挤压量 s -> 稳态接触力）……")
    s_grid = np.linspace(0.0, 1.0, 9)
    cal = force_calibration(s_grid)
    for s, f in zip(s_grid, cal):
        print(f"  s={s:.2f} -> {f:6.2f} N")

    f_lo, f_hi = float(cal.min()), float(cal.max())
    sps = (f_lo + 0.30 * (f_hi - f_lo), f_lo + 0.70 * (f_hi - f_lo),
           f_lo + 0.50 * (f_hi - f_lo))
    print(f"\n力闭环跟踪阶跃设定值（标定可达区间 {f_lo:.1f} ~ {f_hi:.1f} N，"
          f"设定值取区间内 30%/70%/50% 处）……")
    tracks = force_tracking(sps)

    # ---------------- 图
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.3))
    t = np.arange(len(next(iter(rows.values()))["force_series"])) * CTRL_DT
    for name, kp, kd, col in STIFFNESS:
        ax[0].plot(t, rows[name]["disp_series"] * 1000, lw=1.7, color=col,
                   label=f"{name} (kp={kp})")
        ax[1].plot(t, rows[name]["force_series"], lw=1.7, color=col, label=name)
    for a, ti, yl in ((ax[0], "球心位移（扰动 0.10~0.30 s 施加）", "mm"),
                      (ax[1], "球受到的接触力", "N")):
        a.axvspan(0.10, 0.30, color=CLR_GRAY, alpha=0.16)
        a.set_title(ti)
        a.set_xlabel("时间 (s)")
        a.set_ylabel(yl)
        a.legend(fontsize=8.5)

    ax[2].plot(s_grid, cal, "-o", lw=1.8, ms=4, color=CLR_PRIMARY,
               label="静态标定 s → 稳态接触力")
    for i, tr in enumerate(tracks):
        ax[2].plot(tr["t"], tr["force"], lw=1.4, color=CLR_SEQ[i % len(CLR_SEQ)],
                   label=f"闭环 设定 {tr['setpoint']:.0f} N")
        ax[2].axhline(tr["setpoint"], ls=":", lw=1.0,
                      color=CLR_SEQ[i % len(CLR_SEQ)])
    ax[2].set_title("握持力控制（曲线=闭环跟踪，虚线=设定值）")
    ax[2].set_xlabel("时间 (s)／标定: 收紧度 s")
    ax[2].set_ylabel("接触力 (N)")
    ax[2].legend(fontsize=7.5)

    fig.suptitle(f"Shadow Hand 阻抗控制与力控 · 阶跃扰动 {args.force:.1f} N", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "impedance_force.png"))
    plt.close(fig)
    print("\n图 -> results/impedance_force.png")

    with open(os.path.join(RES, "impedance_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(force_n=args.force,
                       response={k: {kk: vv for kk, vv in v.items()
                                     if not kk.endswith("_series")}
                                 for k, v in rows.items()},
                       calibration=dict(s=s_grid.tolist(), force=cal.tolist()),
                       tracking=[dict(setpoint=t["setpoint"], steady=t["steady"],
                                      error=t["error"]) for t in tracks]),
                  f, ensure_ascii=False, indent=2)
    print("指标 -> results/impedance_metrics.json")


if __name__ == "__main__":
    main()
