#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ShadowHandReorientEnv —— 掌内定向旋转任务的 Gymnasium 标准环境
================================================================

任务定义
--------
球（半径 4 cm）被五指包络握在掌窝里。episode 开始时记下球的初始朝向，
随机给定一个目标转角 theta_target ∈ [25°, 75°]，要求把球**相对初始朝向**转过
theta_target（顺时针逆时针都算）。全程球不许掉出掌心。

  * 成功：|theta_now − theta_target| ≤ 10°，且仍有 ≥2 个手部连杆接触球
  * 失败：球掉出掌心（episode 终止 + 惩罚）
  * 超时：1500 步 = 3.0 s

为什么用"测地转角"而不是绕某个固定轴的角度
------------------------------------------
球是均匀球体，绕哪个轴转在几何上等价。theta = 2·acos|⟨q_ref, q_now⟩| ∈ [0, π]
是唯一良定义的标量，而且**天然多模态**——同一个 theta_target 既可以顺时针搓到，
也可以逆时针搓到。这一点在 07 的模仿学习里很关键：它会让"对多模态示范取平均"的
BC/ACT 露怯，而 Diffusion Policy 的多模态建模能力能显示出来。

这也让奖励**不可作弊**：theta 有界于 π，靠高频抖动无法把它刷大，必须真的把球转动。

观测 / 动作
-----------
  obs : 手 24 关节角 + 24 关节速度 + 球在掌心系下的位置(3)/四元数(4)/线速度(3)/角速度(3)
        + theta_now(1) + theta_target(1)   -> 共 63 维
  act : 24 维，∈[-1,1]，映射为 q_des = q_grasp + a · ACTION_SCALE
        （在抓握姿势附近工作，与经典"行波协同"控制器同一工作点，便于公平对比）

与经典控制器的关系
------------------
本环境带了一个 expert_action()，它就是 03 工程里那条"四指行波 + 拇指反摆"协同，
直接以同一动作空间输出。于是"经典控制器 vs 强化学习策略"可以在**完全相同的环境、
相同的观测与动作空间**下对比，这是本工程最重要的一个对照。
"""
from __future__ import annotations

import math
import os

import numpy as np
import mujoco

import gymnasium as gym
from gymnasium import spaces

import hand_common as H

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 常量
BALL_SETTLED = np.array([0.3330, 0.0011, 0.0094])
PALM_NAME = "lh_palm"
BALL_BODY = "ball"
BALL_GEOM = "ball_geom"

ACTION_SCALE = 1.0
CTRL_DT = 0.002
EPISODE_STEPS = 1500
SETTLE_STEPS = 30

# 目标区间由实测能力标定：五指"笼子"会把球推到某个锁定姿态，行波协同的净转角
# 峰值在 47°~118° 之间（取决于初始接触几何）。取 25°~75° 既保证任务非平凡，
# 又留出"经典协同成功率约六成、学习策略可以超过它"的空间。
TARGET_MIN, TARGET_MAX = math.radians(25.0), math.radians(75.0)
SUCCESS_TOL = math.radians(10.0)
# 必须在容差内**连续保持**这么多步才算成功。
# 不加这个 dwell 判定的话，粗暴挥动会把球偶然扫过目标角，
# 实测纯随机策略也能"成功"11/12 —— 那就完全失去区分度了。
SUCCESS_DWELL = 60                       # 0.12 s

DROP_Z = -0.030
MAX_PALM_DIST = 0.16

W_PROGRESS = 20.0
W_SUCCESS = 50.0
W_ALIVE = 0.02
W_CENTER = 1.5
W_ENERGY = 0.0005
W_NO_CONTACT = 0.03
P_DROP = 25.0

RAND_BALL_POS = 0.003
RAND_GRASP = 0.010


# ------------------------------------------------------------------ 工具
def quat_inv(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(q1, float), np.asarray(q2, float))
    return out


def quat_geodesic(q1, q2):
    """两四元数间的测地转角（rad），∈[0, π]。abs 处理四元数双重覆盖。"""
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis])


# ------------------------------------------------------------------ 环境
class ShadowHandReorientEnv(gym.Env):
    """掌内定向旋转（in-hand reorientation）。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, width=480, height=360, randomize=True):
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.randomize = randomize

        self.model = mujoco.MjModel.from_xml_path(H.SCENE)
        self.data = mujoco.MjData(self.model)
        self.ctrl = H.HandController(self.model, self.data)

        m = self.model
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
        jid = m.body_jntadr[bid]
        self.ball_body = bid
        self.ball_qadr = m.jnt_qposadr[jid]
        self.ball_dadr = m.jnt_dofadr[jid]
        self.ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM)
        self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PALM_NAME)

        keys = H.parse_keyframes()
        self.q_grasp = np.asarray(keys["grasp sphere"], float)
        self.q_lo = self.ctrl.lo.copy()
        self.q_hi = self.ctrl.hi.copy()

        self.n_hand = len(self.ctrl.qadr)
        obs_dim = self.n_hand * 2 + 3 + 4 + 3 + 3 + 2
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_hand,), np.float32)

        self._renderer = None
        self._q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        self._theta = 0.0
        self._theta_target = 0.0
        self._cumu_rot = 0.0
        self._palm_ref = BALL_SETTLED.copy()
        self._t = 0
        self._success = False
        self._dropped = False
        self._dwell = 0
        self._min_contacts = 99
        self._sum_contacts = 0.0
        self._n_contact_samples = 0
        self.rng = np.random.default_rng(0)

    # -------------------------------------------------------------- 内部
    def _settle(self, q_des, n):
        for _ in range(n):
            self.ctrl.drive(q_des)
            mujoco.mj_step(self.model, self.data)

    def _ball_contacts(self):
        """球与手部连杆的接触数 / 不同连杆数。

        原实现逐 c 访问 self.data.contact[c]，每次都在 Python 里构造一个
        MjContact 包装对象，3 个接触也要 ~85 µs；改成直接读 contact.geom
        的 (ncon,2) 视图做布尔掩码，约 8 µs。这是每步都调的热路径。
        """
        g = self.data.contact.geom
        if g.shape[0] == 0:
            return 0, 0
        m0 = g[:, 0] == self.ball_geom
        m1 = g[:, 1] == self.ball_geom
        mask = m0 | m1
        if not mask.any():
            return 0, 0
        gb = g[mask]
        other = np.where(m0[mask], gb[:, 1], gb[:, 0])
        bodies = self.model.geom_bodyid[other]
        bodies = bodies[bodies != self.ball_body]
        return int(bodies.size), int(np.unique(bodies).size)

    def _obs(self):
        d = self.data
        q = d.qpos[self.ctrl.qadr].copy()
        qd = d.qvel[self.ctrl.dof].copy()

        p_palm = d.xpos[self.palm].copy()
        R = d.xmat[self.palm].reshape(3, 3).copy()
        q_palm = d.xquat[self.palm].copy()

        p_ball = d.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
        q_ball = d.qpos[self.ball_qadr + 3:self.ball_qadr + 7].copy()
        v_lin = d.qvel[self.ball_dadr:self.ball_dadr + 3].copy()
        v_ang = d.qvel[self.ball_dadr + 3:self.ball_dadr + 6].copy()

        p_rel = R.T @ (p_ball - p_palm)
        q_rel = quat_mul(quat_inv(q_palm), q_ball)

        mid = 0.5 * (self.q_hi + self.q_lo)
        half = 0.5 * (self.q_hi - self.q_lo)
        q_n = (q - mid) / half
        qd_n = np.clip(qd / 10.0, -3.0, 3.0)

        obs = np.concatenate([
            q_n, qd_n,
            p_rel / 0.10,
            q_rel,
            R.T @ v_lin,
            R.T @ v_ang / 10.0,
            [self._theta / math.pi, self._theta_target / math.pi],
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # -------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)

        q0 = self.q_grasp.copy()
        if self.randomize:
            q0 = q0 + self.rng.uniform(-RAND_GRASP, RAND_GRASP, size=q0.shape)
        q0 = np.clip(q0, self.q_lo, self.q_hi)
        d.qpos[self.ctrl.qadr] = q0
        d.qvel[:] = 0.0

        p = BALL_SETTLED.copy()
        if self.randomize:
            p = p + self.rng.uniform(-RAND_BALL_POS, RAND_BALL_POS, size=3)
        axis = self.rng.normal(size=3)
        q_rand = quat_from_axis_angle(axis, self.rng.uniform(0, 2 * math.pi))
        d.qpos[self.ball_qadr:self.ball_qadr + 3] = p
        d.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = q_rand
        mujoco.mj_forward(m, d)

        # 短 settle：让接触稳定，否则初始帧有穿插、力尖峰
        self._settle(q0, SETTLE_STEPS)

        self._q_ref = d.qpos[self.ball_qadr + 3:self.ball_qadr + 7].copy()
        if np.dot(self._q_ref, q_rand) < 0:
            self._q_ref = -self._q_ref
        self._palm_ref = d.xpos[self.palm].copy()
        self._theta = 0.0
        self._cumu_rot = 0.0
        self._t = 0
        self._success = False
        self._dropped = False
        self._dwell = 0
        self._min_contacts = 99
        self._sum_contacts = 0.0
        self._n_contact_samples = 0

        if self.randomize:
            self._theta_target = float(self.rng.uniform(TARGET_MIN, TARGET_MAX))
        else:
            self._theta_target = math.radians(90.0)

        return self._obs(), self._info()

    def step(self, action):
        d, m = self.data, self.model
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        q_des = np.clip(self.q_grasp + a * ACTION_SCALE, self.q_lo, self.q_hi)

        theta_prev = self._theta
        q_prev = d.qpos[self.ball_qadr + 3:self.ball_qadr + 7].copy()

        self.ctrl.drive(q_des)
        mujoco.mj_step(m, d)

        q_now = d.qpos[self.ball_qadr + 3:self.ball_qadr + 7].copy()
        if np.dot(q_now, self._q_ref) < 0:
            q_now = -q_now
        self._theta = quat_geodesic(self._q_ref, q_now)
        self._cumu_rot += quat_geodesic(q_prev, q_now)

        p_now = d.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
        ncon, nbody = self._ball_contacts()
        self._min_contacts = min(self._min_contacts, nbody)
        self._sum_contacts += nbody
        self._n_contact_samples += 1

        # ---------------- 奖励
        err_prev = abs(theta_prev - self._theta_target)
        err_now = abs(self._theta - self._theta_target)
        r = W_PROGRESS * (err_prev - err_now)
        r += W_ALIVE
        r -= W_CENTER * max(0.0, float(np.linalg.norm(p_now - self._palm_ref)) - 0.02)
        r -= W_ENERGY * float(np.sum(d.qvel[self.ctrl.dof] ** 2))

        terminated = False

        # 必须稳定停在目标角：连续保持 SUCCESS_DWELL 步才算成功
        if abs(self._theta - self._theta_target) <= SUCCESS_TOL and nbody >= 2:
            self._dwell += 1
        else:
            self._dwell = 0
        if nbody < 2:
            r -= W_NO_CONTACT

        if self._dwell >= SUCCESS_DWELL:
            r += W_SUCCESS
            terminated = True
            self._success = True

        if p_now[2] < DROP_Z or float(np.linalg.norm(p_now - self._palm_ref)) > MAX_PALM_DIST:
            r -= P_DROP
            terminated = True
            self._dropped = True

        self._t += 1
        info = self._info()
        return self._obs(), float(r), terminated, self._t >= EPISODE_STEPS, info

    def _info(self):
        p_now = self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
        return {
            "theta": self._theta,
            "theta_target": self._theta_target,
            "theta_err_deg": math.degrees(abs(self._theta - self._theta_target)),
            "cumu_rot_deg": math.degrees(self._cumu_rot),
            "net_rot_deg": math.degrees(self._theta),
            "ball_pos": p_now,
            "step": self._t,
            "mean_contacts": self._sum_contacts / max(1, self._n_contact_samples),
            "min_contacts": self._min_contacts,
            "success": self._success,
            "dropped": self._dropped,
        }

    # -------------------------------------------------------------- 专家
    def expert_action(self, t=None):
        """03 工程里的"四指行波 + 拇指反摆"协同，映射到本环境的动作空间。

        与 shadow_hand_demo.roll_synergy 完全一致，只是输出成 [-1,1] 的归一化偏移，
        这样"经典控制器"与"学习策略"共用同一个动作接口，可以公平对比。
        """
        t = self._t * CTRL_DT if t is None else t
        # 参数由 scan_expert.py / scan_expert_trace.py 扫出：
        #   amp=1.0 freq=2.4 wave=1.6（峰值净转角 117.7°，优于 03 原版 0.9/1.2/1.6 的 44°）
        # 波形 sin+0.30·sin2θ 是**非对称**的：推程压得深、回程抬得浅，
        # 回程指尖相对球打滑，于是每周期留下净转动，而对称正弦会正负抵消。
        amp, freq, wave, skew = 1.00, 2.4, 1.6, 0.30
        q = self.q_grasp.copy()
        w = 2 * math.pi * freq
        for k, i0 in enumerate(H.IDX_FINGER_J4):
            th = w * t + wave * k
            s = math.sin(th) + skew * math.sin(2.0 * th)
            q[i0 + 2] += amp * s
            q[i0 + 3] += amp * 0.30 * s
        q[H.IDX_THJ1] += -0.35 * amp * math.sin(w * t)
        return np.clip((q - self.q_grasp) / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------- 渲染
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.distance, cam.azimuth, cam.elevation = 0.42, 118.0, -18.0
        cam.lookat = np.array([0.33, 0.0, 0.0])
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()

    def close(self):
        self._renderer = None


# ------------------------------------------------------------------ 自测
def run_episode(env, policy, render_every=0):
    obs, info = env.reset()
    total, frames = 0.0, []
    while True:
        act = policy(env) if callable(policy) else policy
        obs, r, term, trunc, info = env.step(act)
        total += r
        if render_every and env._t % render_every == 0:
            frames.append(env.render())
        if term or trunc:
            break
    return total, info, frames


def main():
    import argparse
    ap = argparse.ArgumentParser(description="掌内定向旋转环境自测")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    env = ShadowHandReorientEnv()
    print(f"观测空间 {env.observation_space.shape}  动作空间 {env.action_space.shape}")
    print(f"手关节数 {env.n_hand}  episode {EPISODE_STEPS} 步 = {EPISODE_STEPS * CTRL_DT:.1f} s\n")

    rng = np.random.default_rng(0)
    policies = [
        ("随机动作", lambda e: rng.uniform(-1, 1, e.n_hand).astype(np.float32)),
        ("零动作（只保持握持）", lambda e: np.zeros(e.n_hand, np.float32)),
        ("经典行波协同（专家）", lambda e: e.expert_action()),
    ]
    for label, pol in policies:
        rows = []
        for _ in range(args.episodes):
            ret, info, _ = run_episode(env, pol)
            rows.append((ret, info))

        rets = [r for r, _ in rows]

        def m(k):
            return float(np.mean([i[k] for _, i in rows]))

        n_succ = sum(1 for _, i in rows if i.get("success"))
        n_drop = sum(1 for _, i in rows if i.get("dropped"))
        print(f"[{label}]")
        print(f"   回报 {np.mean(rets):8.2f} | 目标 {math.degrees(m('theta_target')):6.1f}°"
              f" | 净转角 {m('net_rot_deg'):6.1f}° | 累计转角 {m('cumu_rot_deg'):7.1f}°")
        print(f"   目标误差 {m('theta_err_deg'):6.1f}° | 平均接触连杆 {m('mean_contacts'):4.2f}"
              f" | 成功 {n_succ}/{len(rows)} | 掉球 {n_drop}/{len(rows)}\n")
    env.close()


if __name__ == "__main__":
    main()
