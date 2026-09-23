"""实验10：细化扫描"四指行波滚球"参数，并统计累计转角（看是否持续单向滚动）。"""
import math
import numpy as np
import mujoco
import hand_common as H

m = mujoco.MjModel.from_xml_path(H.SCENE)
d = mujoco.MjData(m)
K = H.parse_keyframes()
open_24, grasp_24 = K["open hand"], K["grasp sphere"]
CRADLE = (1 - 0.45) * open_24 + 0.45 * grasp_24
LO24 = np.array([m.jnt_range[i][0] for i in range(m.njnt) if m.jnt_type[i] != 0])
HI24 = np.array([m.jnt_range[i][1] for i in range(m.njnt) if m.jnt_type[i] != 0])
DOFS = np.array([m.jnt_dofadr[i] for i in range(m.njnt) if m.jnt_type[i] != 0])
QADRS = np.array([m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_type[i] != 0])

bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
qa = m.jnt_qposadr[m.body_jntadr[bid]]
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

KP = np.r_[np.array([10.0, 10.0]), np.full(22, 2.5)]
KD = np.r_[np.array([0.4, 0.4]), np.full(22, 0.05)]
m.actuator_gainprm[:, 0] = 0.0
m.actuator_biasprm[:, 1] = 0.0
m.actuator_biasprm[:, 2] = 0.0

IDX = [7, 10, 13, 16]
THJ1, THJ5 = 6, 2


def drive(qd):
    tau = KP * (qd - d.qpos[QADRS]) - KD * d.qvel[DOFS]
    d.qfrc_applied[:] = 0
    d.qfrc_applied[DOFS] = tau


def ncon_hand():
    return sum(1 for c in range(d.ncon) if gid in (d.contact[c].geom1, d.contact[c].geom2))


def qang(q0, q1):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


def setup():
    mujoco.mj_resetData(m, d)
    d.qpos[qa:qa + 3] = [0.30, 0.005, 0.08]
    d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(150):
        drive(open_24); mujoco.mj_step(m, d)
    for _ in range(300):
        drive(CRADLE); mujoco.mj_step(m, d)
    for k in range(400):
        t = k / 399; s = t * t * (3 - 2 * t)
        drive((1 - s) * CRADLE + s * grasp_24); mujoco.mj_step(m, d)
    for _ in range(500):
        drive(grasp_24); mujoco.mj_step(m, d)


def run(amp, freq, wave, steps=6000, verbose=True):
    setup()
    def syn(t):
        c = grasp_24.copy()
        for k, i0 in enumerate(IDX):
            s = math.sin(2 * math.pi * freq * t + wave * k)
            c[i0 + 2] += amp * s
            c[i0 + 3] += amp * 0.3 * s
        c[THJ1] += -0.35 * amp * math.sin(2 * math.pi * freq * t)
        return c
    qprev = d.qpos[qa + 3:qa + 7].copy()
    q0 = qprev.copy()
    p0 = d.qpos[qa:qa + 3].copy()
    cum = 0.0
    n_min, n_sum = 99, 0
    for k in range(steps):
        drive(np.clip(syn(k * m.opt.timestep), LO24, HI24))
        mujoco.mj_step(m, d)
        qc = d.qpos[qa + 3:qa + 7]
        cum += qang(qprev, qc); qprev = qc.copy()
        nc = ncon_hand(); n_min = min(n_min, nc); n_sum += nc
    net = qang(q0, d.qpos[qa + 3:qa + 7])
    drop = float(np.linalg.norm(d.qpos[qa:qa + 3] - p0))
    if verbose:
        print(f"amp={amp:.2f} f={freq:.1f} wave={wave:.1f} | 净转角={net:6.1f}° "
              f"累计={cum:7.1f}° 球位移={drop*1000:5.1f}mm 最少接触={n_min} 平均接触={n_sum/steps:.1f}")
    return net, cum, drop, n_min


print("=== 细化扫描（12s 每例）===")
best = None
for amp in [0.6, 0.75, 0.9, 1.05]:
    for freq in [0.8, 1.2]:
        for wave in [1.6, 2.4]:
            net, cum, drop, nmin = run(amp, freq, wave)
            score = cum if (drop < 0.02 and nmin >= 2) else -1
            if best is None or score > best[0]:
                best = (score, amp, freq, wave, net, cum, drop, nmin)
print("\n最佳:", best)
