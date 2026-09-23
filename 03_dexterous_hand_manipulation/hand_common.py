"""Shadow Hand 灵巧手工程公共模块。

包含：
  * 模型 / 关键帧读取
  * ctrl 映射（20 个执行器里有 4 个是"固定肌腱"执行器，不能直接用关节角当 ctrl）
  * 关节力矩控制（本工程不用模型自带的低增益位置执行器，原因见下）
  * 小工具：平滑插值 / 四元数转角 / 接触统计

为什么不用模型自带的位置执行器：
  Menagerie 的 Shadow Hand 用 <position kp="..."> 直接力执行器，kp 只有 0.4~1.5，
  实际握持力只有 1 N 量级，手指一搓就打滑、根本搓不动球。这里把内置执行器增益置零，
  改成 24 个关节的 PD 力矩（qfrc_applied）+ 重力/科氏补偿，刚度完全由我们控制。
  实测：同样一条"四指行波"协同，改用力矩控制后球的掌内转角从 8° 提升到 100°+。
"""
import math
import os
import re
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "shadow_hand_model")
SCENE = os.path.join(MODEL_DIR, "scene_hand.xml")
KEYFILE = os.path.join(MODEL_DIR, "keyframes.xml")

FINGERS = ["FF", "MF", "RF", "LF"]           # 食 / 中 / 无名 / 小指
TIP_BODIES = ["lh_ffdistal", "lh_mfdistal", "lh_rfdistal", "lh_lfdistal", "lh_thdistal"]
# 24 个手关节在 qpos / qvel 里的顺序（由 left_hand.xml 的关节定义顺序决定）
IDX_WRJ2, IDX_WRJ1 = 0, 1
IDX_TH = 2                                   # THJ5..THJ1 => 2..6
IDX_FINGER_J4 = [7, 10, 13, 16]              # FF/MF/RF/LF 的 J4；+1=J3，+2=J2，+3=J1
IDX_THJ5, IDX_THJ4, IDX_THJ3, IDX_THJ2, IDX_THJ1 = 2, 3, 4, 5, 6


# ---------------------------------------------------------------- 读取
def parse_keyframes(path=KEYFILE):
    """keyframes.xml -> {名字: np.array(24 维关节角)}"""
    txt = open(path, "r", encoding="utf-8").read()
    return {m.group(1): np.array([float(v) for v in m.group(2).split()])
            for m in re.finditer(r'<key\s+name="([^"]+)"\s+qpos="([^"]+)"\s*/>', txt)}


def build_ctrl_map(m):
    """每个执行器 -> (关节 qposadr 列表, 系数列表)。支持 joint 与 fixed tendon 两种传动。"""
    cmap = []
    for i in range(m.nu):
        tt = m.actuator_trntype[i]
        tid = m.actuator_trnid[i][0]
        if tt == mujoco.mjtTrn.mjTRN_JOINT:
            cmap.append(([m.jnt_qposadr[tid]], [1.0]))
        elif tt == mujoco.mjtTrn.mjTRN_TENDON:
            adr, num = m.tendon_adr[tid], m.tendon_num[tid]
            qs, cs = [], []
            for w in range(adr, adr + num):
                if m.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT:
                    qs.append(m.jnt_qposadr[m.wrap_objid[w]])
                    cs.append(m.wrap_prm[w])
            cmap.append((qs, cs))
        else:
            cmap.append(([], []))
    return cmap


def qpos_to_ctrl(cmap, q_joints):
    """24 维关节角 -> 20 维 ctrl（肌腱执行器取被耦合关节角之和）。"""
    return np.array([sum(c * q_joints[a] for a, c in zip(qs, cs)) for qs, cs in cmap])


# ---------------------------------------------------------------- 工具
def smoothstep(a, b, t):
    """a -> b 的光滑插值（首尾一阶导为 0，避免速度突变）。"""
    s = t * t * (3 - 2 * t)
    return (1 - s) * np.asarray(a) + s * np.asarray(b)


def quat_angle_deg(q0, q1):
    """两个四元数之间的测地转角（度）。"""
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


def mul_quat(q1, q2):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(q1, float), np.asarray(q2, float))
    return out


# ---------------------------------------------------------------- 控制器
class HandController:
    """把 Shadow Hand 变成"24 关节 PD 力矩 + 重力补偿"的干净接口。"""

    def __init__(self, model, data, kp_finger=2.5, kd_finger=0.05,
                 kp_wrist=10.0, kd_wrist=0.4, gravity_comp=True):
        self.m, self.d = model, data
        # 关掉模型自带的位置执行器（否则会和我们的力矩打架）
        self.m.actuator_gainprm[:, 0] = 0.0
        self.m.actuator_biasprm[:, 1] = 0.0
        self.m.actuator_biasprm[:, 2] = 0.0
        self.cmap = build_ctrl_map(model)
        # 手部关节（排除球的自由关节）
        hand_j = [i for i in range(model.njnt)
                  if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        self.qadr = np.array([model.jnt_qposadr[i] for i in hand_j])
        self.dof = np.array([model.jnt_dofadr[i] for i in hand_j])
        lo = np.array([model.jnt_range[i][0] for i in hand_j])
        hi = np.array([model.jnt_range[i][1] for i in hand_j])
        # 关节角上下限（带一点余量，避免贴边）
        self.lo, self.hi = lo + 1e-4, hi - 1e-4
        kp = np.full(len(hand_j), kp_finger)
        kd = np.full(len(hand_j), kd_finger)
        kp[IDX_WRJ2], kp[IDX_WRJ1] = kp_wrist, kp_wrist
        kd[IDX_WRJ2], kd[IDX_WRJ1] = kd_wrist, kd_wrist
        self.kp, self.kd = kp, kd
        self.gravity_comp = gravity_comp

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def free_builtin(self):
        """让内置位置执行器输出 0 力（把目标设成当前关节角）。"""
        self.d.ctrl[:] = qpos_to_ctrl(self.cmap, self.d.qpos)

    def drive(self, q_des):
        """24 关节 PD 力矩 + 重力/科氏补偿。"""
        q_des = np.clip(np.asarray(q_des, float), self.lo, self.hi)
        q = self.d.qpos[self.qadr]
        tau = self.kp * (q_des - q) - self.kd * self.d.qvel[self.dof]
        if self.gravity_comp:
            tau = tau + self.d.qfrc_bias[self.dof]
        self.d.qfrc_applied[:] = 0.0
        self.d.qfrc_applied[self.dof] = tau
        self.free_builtin()
        return tau
