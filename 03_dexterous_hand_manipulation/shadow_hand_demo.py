#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shadow Hand 灵巧手 · 掌内操纵闭环（抓取 + 掌内滚动 + 卸料）
==========================================================

动作链：张开手掌 -> 掌窝承接料球 -> 五指包络握紧 -> 掌内正向滚动
        -> 掌内反向滚动 -> 停稳持球 -> 翻腕卸料 -> 料球落入接料盘

模型：MuJoCo Menagerie 官方 Shadow Hand E3M5（左手，24 自由度 / 20 执行器）
场景：自建（蓝渐变天空盒 + 棋盘格地面 + 腕部立柱 + 四壁接料盘）

核心技术点
----------
1. **执行器映射**：20 个执行器里有 4 个是"固定肌腱"执行器
   （lh_FFJ0 = lh_FFJ2 + lh_FFJ1，其余三指同理），所以 24 维关节角不能直接当
   20 维 ctrl 用，必须按传动关系映射（hand_common.build_ctrl_map）。

2. **自写关节力矩控制**：模型自带的位置执行器 kp 只有 0.4~1.5（握持力 ~1 N），
   手指一搓就打滑。本工程把内置执行器增益置零，改成 24 关节 PD 力矩 + 重力/科氏
   补偿（qfrc_applied）。同一套协同，球的掌内转角从 8° 提升到 100°+。

3. **掌内滚动协同**：四指屈伸按指序做**行波**（相位差 wave·k），让接触点沿球面
   连续扫过产生滚动；拇指反向小幅摆动提供约束。sign=±1 即正/反向滚动。
   参数由随附的 scan_roll_params.py 网格扫描得到（单段 12 s：净转 ~98.5°、
   累计 ~441°、球在掌内漂移 4.4 mm；正反两段合计累计转角约 900°）。

4. **翻转卸料**：手掌朝上时张开手指，球只会停在掌心（手就是个托盘）；
   所以最后让整只手绕世界 y 轴前倾 20°，球顺势滚出落入接料盘。

5. 球上的三个彩色标记只作视觉用（不参与碰撞），否则球在掌心自转肉眼看不出来。

用法
----
    python shadow_hand_demo.py            # 只输出关键帧 PNG
    python shadow_hand_demo.py --check    # 只跑逻辑并校验，不渲染（快速）
    python shadow_hand_demo.py --viewer   # 实时 3D 窗口
    python shadow_hand_demo.py --mp4      # 额外输出 mp4（可选）
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import mujoco

import hand_common as H

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
MP4 = os.path.join(HERE, "shadow_hand_demo.mp4")

# ------------------------------------------------------------------ 参数
BALL_START = np.array([0.30, 0.005, 0.08])   # 料球初始位置（掌上方）
CRADLE_ALPHA = 0.45                          # 掌窝姿态 = open 与 grasp 的插值系数
KP_FINGER, KD_FINGER = 2.5, 0.05             # 手指 PD 增益
KP_WRIST, KD_WRIST = 10.0, 0.4               # 腕部 PD 增益

ROLL_AMP, ROLL_FREQ, ROLL_WAVE = 0.90, 1.2, 1.6   # 掌内滚动协同参数
DUMP_DEG = 20.0                                   # 卸料前倾角
PIVOT = np.array([0.225, 0.0, 0.012])             # 整手翻转的支点（取在腕部）
CTRL_DT = 0.002
RENDER_EVERY = 20

TRAY_C = np.array([0.47, 0.0, -0.150])


# ------------------------------------------------------------------ 构建
def build_scene():
    model = mujoco.MjModel.from_xml_path(H.SCENE)
    data = mujoco.MjData(model)
    ctrl = H.HandController(model, data, kp_finger=KP_FINGER, kd_finger=KD_FINGER,
                            kp_wrist=KP_WRIST, kd_wrist=KD_WRIST)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    jid = model.body_jntadr[bid]
    ball = dict(body=bid, qadr=model.jnt_qposadr[jid], dadr=model.jnt_dofadr[jid],
                geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_forearm")
    root0 = dict(quat=model.body_quat[root].copy(), pos=model.body_pos[root].copy())

    keys = H.parse_keyframes()
    poses = {"open": keys["open hand"], "grasp": keys["grasp sphere"]}
    poses["cradle"] = (1 - CRADLE_ALPHA) * poses["open"] + CRADLE_ALPHA * poses["grasp"]
    return model, data, ctrl, ball, poses, root, root0


def set_hand_tilt(model, data, root, root0, angle):
    """把整只手绕世界 y 轴前倾 angle（绕 PIVOT 转），模拟机械臂翻腕。"""
    model.body_quat[root] = root0["quat"].copy()
    model.body_pos[root] = root0["pos"].copy()
    q_add = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])
    model.body_quat[root] = H.mul_quat(q_add, root0["quat"])
    mujoco.mj_forward(model, data)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_add)
    model.body_pos[root] = root0["pos"] + (PIVOT - R.reshape(3, 3) @ PIVOT)
    mujoco.mj_forward(model, data)


def reset_ball(model, data, ball, root, root0):
    mujoco.mj_resetData(model, data)
    set_hand_tilt(model, data, root, root0, 0.0)
    data.qpos[ball["qadr"]:ball["qadr"] + 3] = BALL_START
    data.qpos[ball["qadr"] + 3:ball["qadr"] + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def ball_contacts(model, data, ball):
    """球与手部的接触点数 / 接触到的不同连杆数（不含球自身与接料盘）。"""
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


def in_tray(p):
    return (abs(p[0] - TRAY_C[0]) < 0.10 and abs(p[1] - TRAY_C[1]) < 0.10
            and abs(p[2] - (TRAY_C[2] + 0.046)) < 0.06)


# ------------------------------------------------------------------ 协同
def roll_synergy(t, base, sign=+1.0, amp=ROLL_AMP, freq=ROLL_FREQ, wave=ROLL_WAVE, ramp=1.0):
    """四指行波 + 拇指反摆 -> 球在掌心滚动。sign=-1 反向。"""
    q = base.copy()
    a = amp * (min(1.0, t / ramp) if ramp > 0 else 1.0)
    w = sign * 2 * math.pi * freq
    for k, i0 in enumerate(H.IDX_FINGER_J4):
        s = math.sin(w * t + wave * k)
        q[i0 + 2] += a * s
        q[i0 + 3] += a * 0.30 * s
    q[H.IDX_THJ1] += -0.35 * a * math.sin(w * t)
    return q


# ------------------------------------------------------------------ 编排
def build_phases(poses):
    """dict: name / dur / kind(blend|hold|roll) / to / sign / tilt"""
    return [
        dict(name="张开手掌",       dur=1.2, kind="blend", to=poses["open"]),
        dict(name="掌窝预成型",     dur=0.9, kind="blend", to=poses["cradle"]),
        dict(name="料球落入掌窝",   dur=1.6, kind="hold",  to=poses["cradle"]),
        dict(name="五指包络握紧",   dur=1.2, kind="blend", to=poses["grasp"]),
        dict(name="握持稳定",       dur=0.8, kind="hold",  to=poses["grasp"]),
        dict(name="掌内滚动(正向)", dur=6.0, kind="roll",  sign=+1.0),
        dict(name="停顿",           dur=0.6, kind="hold",  to=poses["grasp"]),
        dict(name="掌内滚动(反向)", dur=6.0, kind="roll",  sign=-1.0),
        dict(name="停稳持球",       dur=0.9, kind="hold",  to=poses["grasp"]),
        dict(name="翻腕卸料",       dur=1.5, kind="dump",  to=poses["open"], tilt=DUMP_DEG),
        dict(name="料球落入接料盘", dur=1.8, kind="hold",  to=poses["open"], tilt=DUMP_DEG),
    ]


CAPTURE = {
    ("张开手掌", 0.95):       "01_张开手掌_24自由度",
    ("料球落入掌窝", 0.95):   "02_料球落入掌窝",
    ("握持稳定", 0.92):       "03_五指包络握紧",
    ("掌内滚动(正向)", 0.30): "04_掌内滚动_正向",
    ("掌内滚动(正向)", 0.85): "05_掌内滚动_球已转过一个角度",
    ("掌内滚动(反向)", 0.60): "06_掌内滚动_反向",
    ("停稳持球", 0.90):       "07_稳定持球",
    ("翻腕卸料", 0.90):       "08_翻腕卸料",
    ("料球落入接料盘", 0.95): "09_料球落入接料盘",
}


def main():
    ap = argparse.ArgumentParser(description="Shadow Hand 掌内操纵演示")
    ap.add_argument("--check", action="store_true", help="只跑逻辑并校验，不渲染")
    ap.add_argument("--viewer", action="store_true", help="实时 3D 窗口")
    ap.add_argument("--mp4", action="store_true", help="额外输出 mp4（默认只出关键帧）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    model, data, ctrl, ball, poses, root, root0 = build_scene()
    phases = build_phases(poses)

    renderer = writer = viewer = None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.35, 0.0, -0.01]
    cam.distance = 0.74
    cam.azimuth = 122.0
    cam.elevation = -14.0

    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = \
            cam.distance, cam.azimuth, cam.elevation
    elif not args.check:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        for f in os.listdir(FRAMES_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(FRAMES_DIR, f))
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        if args.mp4:
            import imageio
            writer = imageio.get_writer(MP4, fps=int(1.0 / CTRL_DT / RENDER_EVERY), quality=8)

    reset_ball(model, data, ball, root, root0)
    ctrl.free_builtin()
    cur = poses["open"].copy()
    tilt_now = 0.0

    q_prev = data.qpos[ball["qadr"] + 3:ball["qadr"] + 7].copy()
    cum_rot = 0.0
    min_con = 999
    p_grasp, p_roll_end = None, None
    done, report = [], []

    print(f"模型 shadow_hand | nq={model.nq} nv={model.nv} nu={model.nu} "
          f"手部关节=24 执行器=20（含 4 个固定肌腱）\n")

    step = 0
    for ph in phases:
        name, dur, kind = ph["name"], ph["dur"], ph["kind"]
        n = int(round(dur / CTRL_DT))
        p0 = cur.copy()
        tilt_to = math.radians(ph.get("tilt", 0.0))
        tilt_from = tilt_now
        for k in range(n):
            t = k * CTRL_DT
            if kind == "blend":
                cmd = H.smoothstep(p0, ph["to"], k / max(1, n - 1))
            elif kind == "dump":
                s = k / max(1, n - 1)
                cmd = H.smoothstep(p0, ph["to"], min(1.0, s * 1.5))
            else:
                cmd = roll_synergy(t, p0, sign=float(ph.get("sign", 1.0)))
            if tilt_to != tilt_from or tilt_to != 0.0:
                tilt_now = tilt_from + (tilt_to - tilt_from) * (k / max(1, n - 1))
                set_hand_tilt(model, data, root, root0, tilt_now)
            ctrl.drive(cmd)
            mujoco.mj_step(model, data)

            qc = data.qpos[ball["qadr"] + 3:ball["qadr"] + 7]
            cum_rot += H.quat_angle_deg(q_prev, qc)
            q_prev = qc.copy()
            if kind == "roll":
                min_con = min(min_con, ball_contacts(model, data, ball)[0])

            if viewer is not None:
                viewer.sync()
                time.sleep(CTRL_DT)
            elif renderer is not None:
                for (pn, prog), fname in CAPTURE.items():
                    if pn == name and k == int(n * prog) and fname not in done:
                        renderer.update_scene(data, cam)
                        import imageio
                        imageio.imwrite(os.path.join(FRAMES_DIR, f"{fname}.png"),
                                        renderer.render())
                        done.append(fname)
                        print(f"  [关键帧] {fname}.png")
            if writer is not None and step % RENDER_EVERY == 0:
                renderer.update_scene(data, cam)
                writer.append_data(renderer.render())
            step += 1

        if kind != "roll":
            cur = ph["to"]
        bp = data.qpos[ball["qadr"]:ball["qadr"] + 3].copy()
        if name == "握持稳定":
            p_grasp = bp.copy()
        if name == "停稳持球":
            p_roll_end = bp.copy()
        nc, nb = ball_contacts(model, data, ball)
        report.append((name, bp.copy(), nc, nb))
        print(f"  {name:16s} 球位={np.round(bp, 4)} 接触={nc}(身体{nb})")

    p_end = data.qpos[ball["qadr"]:ball["qadr"] + 3]
    print("\n================ 校验 ================")
    n_grasp = next((nc for nm, _, nc, _ in report if nm == "握持稳定"), 0)
    print(f"抓握期接触点（握持稳定）: {n_grasp} 点握持  (2 指以上同时接触即可稳定夹持)")
    print(f"掌内滚动期最少接触点    : {min_con}   (手指步态中交替接触是机制本身)")
    print(f"掌内滚动累计转角        : {cum_rot:.1f}°")
    if p_grasp is not None and p_roll_end is not None:
        drift = np.linalg.norm(p_roll_end[:2] - p_grasp[:2]) * 1000
        print(f"球在掌内水平漂移        : {drift:.1f} mm  (抓稳 -> 滚动结束后)")
    print(f"卸载后球最终位置        : {np.round(p_end, 4)}")
    print(f"落入接料盘              : {'是' if in_tray(p_end) else '否'}")

    if writer is not None:
        writer.close()
    if viewer is not None:
        viewer.close()
    print(f"\n关键帧 {len(done)} 张 -> {FRAMES_DIR}")
    if args.mp4:
        print(f"mp4 -> {MP4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
