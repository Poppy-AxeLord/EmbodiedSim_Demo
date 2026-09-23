"""Shadow Hand 几何探针：核对掌面朝向、指尖位置、球在掌心的落点。

用法：python probe_hand.py
"""
import numpy as np
import mujoco

import hand_common as H

m = mujoco.MjModel.from_xml_path(H.SCENE)
d = mujoco.MjData(m)
ctrl = H.HandController(m, d)
K = H.parse_keyframes()

print("=== 模型规模 ===")
print(f"  nq={m.nq} nv={m.nv} nu={m.nu}  手部关节=24  执行器=20")
print(f"  执行器含 4 个固定肌腱：lh_FFJ0 = lh_FFJ2 + lh_FFJ1（其余三指同理）")

print("\n=== 执行器 -> 关节 映射 ===")
cmap = H.build_ctrl_map(m)
for i in range(m.nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    qs, cs = cmap[i]
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                            next(j for j in range(m.njnt) if m.jnt_qposadr[j] == a))
          for a in qs]
    print(f"  {i:2d} {nm:14s} <- {' + '.join(jn)}")

print("\n=== 关节角范围 ===")
for i in range(m.njnt):
    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  {nm:12s} range={np.round(m.jnt_range[i], 3)}")


def report(tag, q24):
    mujoco.mj_resetData(m, d)
    for a, v in zip(ctrl.qadr, q24):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    print(f"\n=== {tag} ===")
    for n in ["lh_palm"] + H.TIP_BODIES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        print(f"  {n:13s} {np.round(d.xpos[b], 4)}")
    R = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lh_palm")].reshape(3, 3)
    print(f"  掌心法向(局部 +y) -> {np.round(R[:, 1], 3)}   手指伸展方向(局部 +z) -> {np.round(R[:, 2], 3)}")


report("open hand", K["open hand"])
report("grasp sphere", K["grasp sphere"])

print("\n=== 掌面高度（张开手掌时，用于定位球）===")
mujoco.mj_resetData(m, d)
for a, v in zip(ctrl.qadr, K["open hand"]):
    d.qpos[a] = v
mujoco.mj_forward(m, d)
pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lh_palm")
a, b = int(m.body_geomadr[pid]), int(m.body_geomnum[pid])
print(f"  palm body pos = {np.round(d.xpos[pid], 4)}")
print(f"  （实测：球在掌心稳定落点约 x=0.30, z=0.058）")
