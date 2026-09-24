#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境设计探针：确定"球稳定落在掌窝里"的初始位姿，并实测仿真吞吐。

为什么要这一步：
  强化学习每个 episode 都要 reset。如果每次 reset 都靠"让球从上方落下再等它稳"，
  就要白跑约 500 个仿真步，训练时这笔开销很可观。这里先把稳定位姿测出来写成常量，
  reset 时直接摆上去，只用很短的 settle 步数（50 步）消除穿插。

同时实测 MuJoCo 步进吞吐，用来估算 CPU 上能训多少步。
"""
import os
import sys
import time

import numpy as np
import mujoco

import hand_common as H

HERE = os.path.dirname(os.path.abspath(__file__))

BALL_DROP = np.array([0.30, 0.005, 0.08])   # 与 03 演示一致的落球点
CRADLE_ALPHA = 0.45
CTRL_DT = 0.002


def build():
    model = mujoco.MjModel.from_xml_path(H.SCENE)
    data = mujoco.MjData(model)
    ctrl = H.HandController(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    jid = model.body_jntadr[bid]
    ball = dict(body=bid, qadr=model.jnt_qposadr[jid], geom=mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_palm")
    keys = H.parse_keyframes()
    poses = {"open": keys["open hand"], "grasp": keys["grasp sphere"]}
    poses["cradle"] = (1 - CRADLE_ALPHA) * poses["open"] + CRADLE_ALPHA * poses["grasp"]
    return model, data, ctrl, ball, palm, poses


def drive(ctrl, q_des, n_steps, model, data):
    tau_prev = np.zeros(24)
    for _ in range(n_steps):
        tau_prev = ctrl.drive(q_des)
        mujoco.mj_step(model, data)
    return tau_prev


def contacts_with_hand(model, data, ball):
    n, bodies = 0, set()
    for c in range(data.ncon):
        con = data.contact[c]
        if ball["geom"] in (con.geom1, con.geom2):
            other = con.geom1 if con.geom2 == ball["geom"] else con.geom2
            b = int(model.geom_bodyid[other])
            if b != ball["body"]:
                n += 1
                bodies.add(b)
    return n, len(bodies)


def main():
    model, data, ctrl, ball, palm, poses = build()
    print(f"模型: nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"palm body id = {palm}\n")

    # ---------- 1) 让球从上方落下，分别落在 cradle / grasp 姿态里 ----------
    settled = {}
    for name in ("cradle", "grasp"):
        mujoco.mj_resetData(model, data)
        data.qpos[ball["qadr"]:ball["qadr"] + 3] = BALL_DROP
        data.qpos[ball["qadr"] + 3:ball["qadr"] + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        q = poses["cradle"]
        drive(ctrl, q, int(0.8 / CTRL_DT), model, data)
        q = poses[name]
        drive(ctrl, q, int(2.5 / CTRL_DT), model, data)

        p = data.qpos[ball["qadr"]:ball["qadr"] + 3].copy()
        quat = data.qpos[ball["qadr"] + 3:ball["qadr"] + 7].copy()
        ncon, nbody = contacts_with_hand(model, data, ball)
        palm_p = data.xpos[palm].copy()
        rel = p - palm_p
        settled[name] = (p, quat)
        print(f"[{name}] 球位姿 = {np.round(p, 4)}")
        print(f"       球四元数 = {np.round(quat, 4)}")
        print(f"       相对掌心 = {np.round(rel, 4)}  |rel| = {np.linalg.norm(rel) * 1000:.1f} mm")
        print(f"       接触点 = {ncon}，接触连杆数 = {nbody}\n")

    # ---------- 2) 掌心坐标系（用于把球位姿转成"掌内相对位姿"）----------
    mujoco.mj_resetData(model, data)
    drive(ctrl, poses["grasp"], 200, model, data)
    print("=== grasp 姿态下的掌心坐标系 ===")
    print("  palm pos  =", np.round(data.xpos[palm], 4))
    print("  palm xmat =", np.round(data.xmat[palm].reshape(3, 3), 3), sep="\n")

    # ---------- 3) 仿真吞吐实测 ----------
    mujoco.mj_resetData(model, data)
    q = poses["grasp"]
    mujoco.mj_forward(model, data)
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        ctrl.drive(q)
        mujoco.mj_step(model, data)
    dt = time.perf_counter() - t0
    print(f"\n=== 仿真吞吐 ===")
    print(f"  含 PD 力矩控制：{n / dt:,.0f} steps/s  ({dt * 1e3 / n:.3f} ms/step)")


if __name__ == "__main__":
    main()
