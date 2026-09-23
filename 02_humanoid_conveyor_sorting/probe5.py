"""probe5: 校验「调整后的取件工位 / 出料槽」位置是否仍在可达区内。"""
import numpy as np
import mujoco
import humanoid_conveyor as H

model, data, items, hand_id, jinfo, lq, ld, llo, lhi, wq, wd, wlo, whi, ci, cb = H.build_scene()
for _ in range(400):
    q = data.qpos[7:7 + H.NU]; qd = data.qvel[6:6 + H.NU]
    tau = H.Kp * (H.HOME - q) - H.Kd * qd
    for i, jn in enumerate(H.JOINT_NAMES):
        tau[i] += data.qfrc_bias[jinfo[jn][1]]
    tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for i, jn in enumerate(H.JOINT_NAMES):
        data.ctrl[jinfo[jn][2]] = tau[i]
    mujoco.mj_step(model, data)
qpos_ref = data.qpos.copy()
print("settled pelvis z =", round(data.qpos[2], 4))

tests = []
for x in (0.46, 0.50, 0.54):
    for y in (-0.35, -0.20, 0.0, 0.20, 0.35):
        tests.append((x, y, 1.04, "belt"))
for x in (0.20, 0.22, 0.26):
    for y in (-0.35, -0.20, 0.0, 0.20, 0.35):
        tests.append((x, y, 1.04, "chute"))
for x in (0.20, 0.26, 0.50):
    for y in (-0.45, 0.45):
        tests.append((x, y, 1.04, "far"))

for x, y, z, tag in tests:
    yaw, e = H.plan_yaw(model, mujoco.MjData(model), hand_id, np.array([x, y, z]),
                        qpos_ref, wq, wd, wlo, whi)
    flag = "OK " if e < 0.01 else ("~  " if e < 0.03 else "BAD")
    print(f"  {flag} {tag:5} ({x:.2f},{y:+.2f},{z:.2f}) err={e*100:5.2f}cm yaw={np.degrees(yaw):+6.1f}deg")
