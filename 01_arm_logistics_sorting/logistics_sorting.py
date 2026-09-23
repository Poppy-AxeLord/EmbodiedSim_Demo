"""
Unitree/Panda 物流分拣单元 — 视觉→抓取→放置 闭环演示 (MuJoCo)
============================================================
一个完整的「视觉→抓取→放置」闭环：**物流分拣工位**。

场景（均有真实模型与场景）：
  * Franka Emika Panda 7-DoF 机械臂（MuJoCo Menagerie 官方模型）
  * 一条传送带，持续送入彩色料箱（box）
  * 三个按颜色分类的料箱（红 / 蓝 / 绿）

动作闭环：
  * 传送带把料箱送到抓取点 →
  * "感知"模块读取料箱颜色（此处用真值占位，对应真实系统的检测/分割/分类）→
  * 机械臂 DLS 逆运动学定位 + 力控夹爪抓取 →
  * 按颜色分拣，搬运并放入对应料箱。

控制架构（可逐层讲解）：
  * 任务空间：用阻尼最小二乘 IK 解末端位姿（雅可比取自物理模型，零手工运动学）
  * 关节层：位置执行器跟踪 IK 解；夹爪为力控弹簧式执行器（约 12N 夹持力）
  * "感知"层：classify() 返回颜色 -> 路由到对应料箱（真实系统即 RGB-D + 6D 位姿 + 分类器）

运行：
  python logistics_sorting.py              # 离线渲染关键帧（图片）
  python logistics_sorting.py --viewer     # 实时 3D 窗口
  python logistics_sorting.py --mp4 out.mp4  # 可选：额外输出视频
依赖：mujoco, numpy, imageio, opencv-python-headless
"""
import os
import argparse
import time

import numpy as np
import mujoco
import imageio

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

MODEL_DIR = r"D:\embodied_ai\assets\menagerie\models\franka_emika_panda-3d2262eeb81ecec1"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先用随仓库打包的模型（桌面/GitHub 副本开箱即用），否则回退到本机 menagerie 路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_HERE, "panda_model")
if os.path.exists(_LOCAL_MODEL):
    MODEL_DIR = _LOCAL_MODEL

# ---- 几何参数（依据 Panda 可达包络设计）----
ITEM_HALF = 0.025          # 料箱半边长 5cm
BELT_TOP = 0.40            # 工作台/传送带台面高度
ITEM_Z = BELT_TOP + ITEM_HALF
PICK_Y = -0.28             # 传送带料道 y
PICK_X = 0.56              # 抓取点 x
BELT_SPEED = 0.08          # 传送带速度 (m/s)
BIN_Y = 0.28               # 料箱排布 y
BIN_XS = [0.34, 0.46, 0.58]   # 红/蓝/绿 料箱 x
BIN_INNER_BOTTOM = 0.406
BIN_DROP_Z = BIN_INNER_BOTTOM + ITEM_HALF + 0.01

# 物料清单（颜色 -> 路由料箱）；最后再放一个红色，演示"同色归箱"
ITEM_COLORS = ["red", "blue", "green", "red"]
CATEGORIES = ["red", "blue", "green"]
ITEM_RGB = {
    "red": (0.85, 0.20, 0.18),
    "blue": (0.15, 0.45, 0.85),
    "green": (0.20, 0.70, 0.30),
}
# 初始在传送带上的间距摆位（最高 x 的最先到达抓取点）
SPAWN_X = [0.56, 0.42, 0.28, 0.14]

GRIP_OPEN = 255.0
GRIP_CLOSE = 30.0
CTRL_DT = 0.02
N_SUBSTEPS = 10
DT = CTRL_DT / N_SUBSTEPS

# 末端朝向：夹爪朝下、两指沿世界 Y 张开（桌面俯视抓取）
EE_DOWN_R = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])

READY = np.array([0.45, 0.0, 0.72])   # 机械臂待机位姿（安全高位）


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------
def add_bin(spec, x, y, rgb, name):
    """在 (x,y) 处放一个开口料箱（底 + 4 壁），颜色用于类别区分。"""
    b = spec.worldbody.add_body(name=name, pos=[x, y, 0.0])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.006],
               pos=[0, 0, 0.403], rgba=list(rgb) + [0.55], friction=[1, 0.05, 0.001])
    cz = 0.406 + 0.03
    for dx in (0.05, -0.05):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, 0.05, 0.03],
                   pos=[dx, 0, cz], rgba=list(rgb) + [0.45])
    for dy in (0.05, -0.05):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.003, 0.03],
                   pos=[0, dy, cz], rgba=list(rgb) + [0.45])


def build_scene():
    os.chdir(MODEL_DIR)
    spec = mujoco.MjSpec.from_file("scene.xml")

    # 工作台
    spec.worldbody.add_geom(
        name="bench", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.34, 0.45, BELT_TOP / 2], pos=[0.46, 0.0, BELT_TOP / 2],
        rgba=[0.32, 0.33, 0.38, 1.0], friction=[1.0, 0.05, 0.001])

    # 传送带（料道）
    spec.worldbody.add_geom(
        name="belt", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.33, 0.17, BELT_TOP / 2], pos=[0.45, PICK_Y - 0.01, BELT_TOP / 2 + 0.005],
        rgba=[0.18, 0.18, 0.20, 1.0], friction=[1.0, 0.05, 0.001])
    # 传送带中心黄色标线（纯视觉）
    spec.worldbody.add_geom(
        name="belt_marker", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.30, 0.004, 0.001], pos=[0.45, PICK_Y, BELT_TOP + 0.004],
        rgba=[0.95, 0.78, 0.12, 0.85], contype=0, conaffinity=0)

    # 三个分类料箱
    for bx, col in zip(BIN_XS, CATEGORIES):
        add_bin(spec, bx, BIN_Y, ITEM_RGB[col], f"bin_{col}")

    # 物料：彩色料箱（自由刚体，运行时由脚本驱动其位姿）
    items = []
    for i, col in enumerate(ITEM_COLORS):
        b = spec.worldbody.add_body(name=f"item{i}", pos=[SPAWN_X[i], PICK_Y, ITEM_Z])
        b.add_freejoint()
        b.add_geom(name=f"item{i}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[ITEM_HALF, ITEM_HALF, ITEM_HALF],
                   rgba=list(ITEM_RGB[col]) + [1.0], mass=0.05,
                   friction=[1.0, 0.05, 0.001])

    # 手掌上的抓取点 site（IK 跟踪的就是它）
    spec.body("hand").add_site(name="grasp", pos=[0.0, 0.0, 0.103],
                               size=[0.006, 0.0, 0.0], rgba=[0.1, 0.6, 1.0, 0.8])

    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 1024

    # 增强夹爪闭合力（与项目 05 一致：约 12N 夹持力，抓得稳）
    grip_act = spec.actuator("actuator8")
    grip_act.biasprm[1] = -600.0
    grip_act.gainprm[0] = 600.0 * 0.04 / 255.0

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    item_info = []
    for i, col in enumerate(ITEM_COLORS):
        bid = model.body(f"item{i}").id
        qadr = model.jnt_qposadr[model.body(f"item{i}").jntadr[0]]
        item_info.append({
            "idx": i, "color": col,
            "bin_idx": CATEGORIES.index(col),
            "body_id": bid, "qadr": qadr,
            "state": "onbelt", "spawn_x": SPAWN_X[i], "x": SPAWN_X[i],
        })
    site_id = model.site("grasp").id
    return model, data, item_info, site_id


# ---------------------------------------------------------------------------
# 逆运动学：阻尼最小二乘（DLS）
# ---------------------------------------------------------------------------
def mat_to_rotvec(R):
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= (2.0 * np.sin(theta))
    return theta * axis


def solve_ik(model, ik_data, site_id, target_pos, target_R, q_init, q_lo, q_hi, iters=40):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q = np.array(q_init, dtype=float).copy()
    for _ in range(iters):
        ik_data.qpos[:7] = q
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)
        err_p = target_pos - ik_data.site_xpos[site_id]
        R_cur = ik_data.site_xmat[site_id].reshape(3, 3)
        err_r = mat_to_rotvec(target_R @ R_cur.T)
        err = np.concatenate([err_p, err_r])
        if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-3:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, :7]
        lam2 = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err)
        dq = np.clip(dq, -0.5, 0.5)
        q = np.clip(q + 0.4 * dq, q_lo, q_hi)
    ik_data.qpos[:7] = q
    mujoco.mj_kinematics(model, ik_data)
    mujoco.mj_comPos(model, ik_data)
    pos_err = np.linalg.norm(target_pos - ik_data.site_xpos[site_id])
    return q, pos_err


def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def classify(item):
    """感知模块：返回料箱颜色（此处用真值占位；真实系统为 RGB-D 检测/分割/分类）。"""
    return item["color"]


def build_phases(item):
    """单个料箱的抓取-放置相位序列。"""
    pick_pos = np.array([PICK_X, PICK_Y, ITEM_Z])
    bx = BIN_XS[item["bin_idx"]]
    bin_drop = np.array([bx, BIN_Y, BIN_DROP_Z])
    bin_above = np.array([bx, BIN_Y, BIN_DROP_Z + 0.15])
    return [
        ("Approach", pick_pos + np.array([0, 0, 0.12]), GRIP_OPEN, 0.8),
        ("Descend", pick_pos, GRIP_OPEN, 0.7),
        ("Grasp", pick_pos, GRIP_CLOSE, 0.5),
        ("Lift", pick_pos + np.array([0, 0, 0.18]), GRIP_CLOSE, 0.7),
        ("Move to bin", bin_above, GRIP_CLOSE, 1.0),
        ("Place into bin", bin_drop, GRIP_CLOSE, 0.7),
        ("Release", bin_drop, GRIP_OPEN, 0.4),
        ("Retreat", bin_above, GRIP_OPEN, 0.6),
    ]


def set_item_pose(data, it, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    qa = it["qadr"]
    data.qpos[qa:qa + 3] = pos
    data.qpos[qa + 3:qa + 7] = quat
    data.qvel[qa:qa + 6] = 0.0


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Panda 物流分拣演示")
    ap.add_argument("--viewer", action="store_true", help="实时 3D 窗口")
    ap.add_argument("--mp4", default=None,
                    help="可选：视频输出路径。默认不生成视频，只输出关键帧图片")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    model, data, items, site_id = build_scene()
    ik_data = mujoco.MjData(model)
    q_lo = model.jnt_range[:7, 0]
    q_hi = model.jnt_range[:7, 1]

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0.0, 0.42]
    cam.distance = 2.3
    cam.azimuth = 115.0
    cam.elevation = -12.0

    renderer = None
    viewer = None
    writer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation
    else:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        if args.mp4:
            writer = imageio.get_writer(args.mp4, fps=int(1.0 / CTRL_DT / 2), quality=8)

    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))

    # 状态机
    belt_running = True
    belt_offset = 0.0
    current = None
    phases = None
    phase_idx = 0
    phase_t = 0.0
    phase_start = None
    phase_target = None
    ee_goal = data.site_xpos[site_id].copy()
    label = "初始化"
    step = 0
    placed_count = 0
    keyframes = []   # (label, frame)
    captured_phase = False

    max_iter = 6000
    print("running logistics sorting demo ...")
    for it in range(max_iter):
        # 决定当前目标
        if current is None:
            if belt_running:
                onbelt = [x for x in items if x["state"] == "onbelt"]
                if onbelt:
                    front = max(onbelt, key=lambda a: a["x"])
                    if front["x"] >= PICK_X:
                        # 到达抓取点 -> 开始抓取
                        current = front
                        belt_running = False
                        phases = build_phases(current)
                        phase_idx = 0
                        phase_t = 0.0
                        phase_start = ee_goal.copy()
                        phase_target = phases[0][1]
                        label = f"Grasping {classify(current)} package"
                    else:
                        label = "Conveyor feeding..."
                else:
                    label = "Sorting complete"
            else:
                label = "Sorting complete"

        if current is None:
            target = READY
            grip = GRIP_OPEN
            name = "待机"
        else:
            name, tgt, grip, dur = phases[phase_idx]
            target = tgt
            # 相位插值
            frac = smoothstep(phase_t / dur)
            ee_goal = phase_start + frac * (phase_target - phase_start)
            label = f"{name} ({classify(current)} -> {CATEGORIES[current['bin_idx']]} bin)"

        # IK 求解并下发
        q_sol, pos_err = solve_ik(model, ik_data, site_id, ee_goal, EE_DOWN_R,
                                  data.qpos[:7], q_lo, q_hi)
        data.ctrl[:7] = q_sol
        data.ctrl[7] = grip

        # 物理步进 + 物料/传送带驱动
        for _ in range(N_SUBSTEPS):
            if belt_running:
                belt_offset += BELT_SPEED * DT
            site_pos = data.site_xpos[site_id]
            for it2 in items:
                if it2["state"] == "onbelt":
                    it2["x"] = it2["spawn_x"] + belt_offset
                    set_item_pose(data, it2, [it2["x"], PICK_Y, ITEM_Z])
                elif it2["state"] == "held":
                    set_item_pose(data, it2, site_pos)
            mujoco.mj_step(model, data)

        # 相位推进
        if current is not None:
            phase_t += CTRL_DT
            if phase_t >= phases[phase_idx][3]:
                if name == "Grasp":
                    current["state"] = "held"
                elif name == "Release":
                    current["state"] = "placed"
                    placed_count += 1
                phase_idx += 1
                captured_phase = False
                if phase_idx >= len(phases):
                    current = None
                    belt_running = True
                    ee_goal = data.site_xpos[site_id].copy()
                else:
                    phase_start = ee_goal.copy()
                    phase_target = phases[phase_idx][1]
                    phase_t = 0.0

        # 渲染 / 记录
        if viewer is not None:
            viewer.sync()
            time.sleep(CTRL_DT)
        else:
            renderer.update_scene(data, cam)
            frame = renderer.render()
            if HAVE_CV2:
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis, (10, 10), (args.width - 10, 46), (0, 0, 0), -1)
                cv2.putText(vis, label, (22, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            if writer is not None and step % 2 == 0:
                writer.append_data(frame)
            # 关键帧采集（每个事件只采一帧，最多 6 张）
            if len(keyframes) < 6 and not captured_phase:
                if step == 0:
                    keyframes.append(("01_传送带送料", frame.copy()))
                    captured_phase = True
                elif name == "Grasp" and current is not None and current["state"] == "onbelt":
                    keyframes.append((f"0{len(keyframes)+1}_抓取{classify(current)}", frame.copy()))
                    captured_phase = True
                elif name == "Release" and current is not None and current["state"] == "held":
                    keyframes.append((f"0{len(keyframes)+1}_放入{current['color']}箱", frame.copy()))
                    captured_phase = True

        step += 1
        if placed_count >= len(items) and current is None:
            # 收尾几帧
            if renderer is not None:
                for _ in range(30):
                    renderer.update_scene(data, cam)
                    fr = renderer.render()
                    if HAVE_CV2:
                        vis = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(vis, (10, 10), (args.width - 10, 46), (0, 0, 0), -1)
                        cv2.putText(vis, "Sorting complete", (22, 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                        fr = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    if writer is not None:
                        writer.append_data(fr)
            break

    # 保存关键帧
    for lab, fr in keyframes:
        imageio.imwrite(os.path.join(OUT_DIR, "frames", f"{lab}.png"), fr)

    if writer is not None:
        writer.close()
        renderer.close()
    if viewer is not None:
        viewer.close()

    # 校验：每个料箱是否落入对应颜色的料箱
    print("\n分拣结果校验：")
    all_ok = True
    for it in items:
        p = data.body(f"item{it['idx']}").xpos
        bx = BIN_XS[it["bin_idx"]]
        d = float(np.sqrt((p[0] - bx) ** 2 + (p[1] - BIN_Y) ** 2))
        ok = (d < 0.08) and (p[2] < 0.55)
        all_ok = all_ok and ok
        print(f"  item{it['idx']} {it['color']:>5} -> 目标{CATEGORIES[it['bin_idx']]}箱 "
              f"实际({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) 偏差{d*100:.1f}cm "
              f"{'OK' if ok else 'WRONG'}")
    print(f"DONE. 已分拣 {placed_count}/{len(items)} 个料箱，"
          f"分拣正确性={'全部正确' if all_ok else '存在错误'}。"
          f"关键帧 -> {frames_dir}" + (f"，mp4 -> {args.mp4}" if args.mp4 else ""))


if __name__ == "__main__":
    main()
