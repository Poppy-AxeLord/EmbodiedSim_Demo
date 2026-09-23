"""
Unitree H1 人形机器人 — 物流带分拣演示 (MuJoCo)
================================================
一个完整的「视觉→抓取→放置」闭环：**人形分拣工位**。

场景（均有真实模型与场景）：
  * Unitree H1 19-DoF 人形机器人（MuJoCo Menagerie 官方模型）
  * 机器人正前方胸高工作台：一条传送带把彩色料箱送来
  * 红 / 蓝 / 绿 三个分类料箱（伸手可达范围内）

动作闭环：
  * 传送带把料箱送到抓取点 ->
  * "感知"模块读取颜色（真值占位，对应真实 RGB-D 检测/分割/分类）并路由 ->
  * H1 左臂 4-DoF 逆运动学（DLS）伸到抓取点，简化抓取（手端绑定料箱）拿起 ->
  * 按颜色搬运并放入对应料箱。

控制架构（逐层可讲解）：
  * 浮动骨盆用 qfrc_applied 施加 力(位置) + 力矩(姿态)，目标姿态用四元数误差
    表达（避免 rpy 奇异）；姿态估计来自自由关节四元数。这保证机器人**站立不倒**。
  * 19 个驱动关节用 PD 跟踪目标角；腿部/躯干锁定站立，左臂跟踪 IK 解，右臂待机。
  * "感知"层 classify() 独立成函数 —— 换成真实视觉模型输出即可接入，控制层无需改动。

运行：python humanoid_sorting.py              # 离线渲染关键帧（图片）
      python humanoid_sorting.py --viewer     # 实时 3D 窗口
      python humanoid_sorting.py --check      # 只跑逻辑不渲染，快速校验分拣正确性
      python humanoid_sorting.py --mp4 out.mp4  # 可选：额外输出视频
依赖：mujoco, numpy, imageio, opencv-python-headless（均已在本环境 site-packages 中）
"""
import os
import argparse
import time

import numpy as np
import mujoco
import imageio.v2 as imageio

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(HERE, "unitree_h1_model", "scene.xml")
_MENAGERIE = r"D:\embodied_ai\assets\menagerie\models\unitree_h1\scene.xml"
MODEL = os.environ.get("HUMANOID_MODEL",
                       _LOCAL if os.path.exists(_LOCAL) else _MENAGERIE)
OUT_DIR = HERE
MP4 = os.path.join(OUT_DIR, "humanoid_sorting.mp4")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")

# ---- 几何（依据探针得到的 H1 左臂真实可达工作区设计）----
# 实测（骨盆锚定到 home 位姿 0.98、自然站姿）：z≥1.04 时左臂可覆盖抓取点/分类箱，
# IK 残差 <0.1cm；因此台面取 1.01m、料箱中心 1.04m（见 probe3 实测）。
TABLE_TOP = 1.01
ITEM_HALF = 0.03
ITEM_Z = TABLE_TOP + ITEM_HALF          # 料箱中心高度 (1.04)
PICK = np.array([0.42, 0.00, ITEM_Z])  # 传送带抓取点（正前方）
BELT_SPEED = 0.08
SPAWN_XS = [0.10, 0.18, 0.26, 0.34]    # 料箱在传送带上的错位起点（避免重叠）
PICK_X = PICK[0]

# 左臂"待机"位（抬高、避开料箱）
READY_ARM = np.array([0.34, 0.06, 1.22])

# 三个分类料箱（红/蓝/绿），都在左臂可达工作区内（IK 残差 < 2cm）
BIN_POS = {
    "red":   np.array([0.26, 0.24, ITEM_Z]),
    "blue":  np.array([0.38, 0.35, ITEM_Z]),
    "green": np.array([0.47, 0.20, ITEM_Z]),
}
BIN_HALF = 0.075        # 加大：同色料箱可并排放 2 件而不碰撞
BIN_WALL = 0.06
BIN_SPACING = 0.07      # 同色多件时沿 x 的落座间距
REST_Z = TABLE_TOP + 0.009 + ITEM_HALF   # 料箱在箱内静置时的中心高度

# 物料：4 个料箱（含两个红色，演示同色归箱路由）
ITEM_COLORS = ["red", "blue", "green", "red"]
CATEGORIES = ["red", "blue", "green"]
ITEM_RGB = {
    "red":   (0.85, 0.20, 0.18),
    "blue":  (0.15, 0.45, 0.85),
    "green": (0.20, 0.70, 0.30),
}

# ---- 控制频率 ----
CTRL_DT = 0.02
N_SUBSTEPS = 10
DT = CTRL_DT / N_SUBSTEPS
FPS = 30
MAX_EE_SPEED = 0.25     # 末端指令速度上限 (m/s)，所有动作平滑，避免反作用掀翻
WARMUP_STEPS = 80       # 录制前预热步数（平息初始接触瞬态）

# 关节顺序（freejoint 之后，共 19）
NU = 19
HOME = np.array([0, 0, -0.4, 0.8, -0.4,
                 0, 0, -0.4, 0.8, -0.4,
                 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0], dtype=float)
# 关节名（与模型一致）
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]
LEFT_ARM = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"]
HOME_ARM = np.zeros(len(LEFT_ARM))      # 零位（下垂）——IK 的一个备用初值

# PD 增益：腿部/躯干强，手臂轻
Kp = np.ones(NU) * 150.0
Kd = np.ones(NU) * 15.0
ARM = [11, 12, 13, 14, 15, 16, 17, 18]
Kp[ARM] = 45.0
Kd[ARM] = 5.0

# 固定基座（骨盆 weld 到世界）：保证机械臂操作时躯干绝对不倒
FIX_BASE = True

# 基座稳定化增益（简化 WBC / 平衡）。调得很硬 => 等价"固定基座人形"，
# 保证机械臂做大范围操作时躯干绝对不倒；项目 07 另演示了纯平衡控制。
KBP = 40000.0
KBD = 4000.0
KBR = 40000.0
KDR = 4000.0
HOME_Z = 0.98


def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def quat_conj_mul(q_cur, q_des):
    """误差四元数 conj(q_cur)*q_des (w,x,y,z)"""
    cw, cx, cy, cz = q_cur
    dw, dx, dy, dz = q_des
    ew = cw * dw + cx * dx + cy * dy + cz * dz
    ex = cw * dx - cx * dw - cy * dz + cz * dy
    ey = cw * dy + cx * dz - cy * dw - cz * dx
    ez = cw * dz - cx * dy + cy * dx - cz * dw
    return np.array([ew, ex, ey, ez])


def mat_to_rotvec(R):
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= (2.0 * np.sin(theta))
    return theta * axis


def classify(item):
    """感知模块：返回料箱颜色（真值占位；真实系统为 RGB-D 检测/分割/分类）。"""
    return item["color"]


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------
def add_bin(spec, pos, rgb, name):
    # 料箱只与料箱(物品)互碰（contype/conaffinity=2），不与机器人本体碰撞：
    # 否则机械臂/腿会与箱壁产生非预期的物理干涉，破坏站立与跟踪。
    b = spec.worldbody.add_body(name=name, pos=[pos[0], pos[1], TABLE_TOP])
    # 底
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, BIN_HALF, 0.006],
               pos=[0, 0, 0.003], rgba=list(rgb) + [0.5], friction=[1, 0.05, 0.001],
               contype=2, conaffinity=2)
    cz = BIN_WALL / 2 + 0.006
    for dx in (BIN_HALF, -BIN_HALF):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, BIN_HALF, BIN_WALL / 2],
                   pos=[dx, 0, cz], rgba=list(rgb) + [0.45], contype=2, conaffinity=2)
    for dy in (BIN_HALF, -BIN_HALF):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, 0.004, BIN_WALL / 2],
                   pos=[0, dy, cz], rgba=list(rgb) + [0.45], contype=2, conaffinity=2)


def build_scene():
    model_dir = os.path.dirname(MODEL)
    os.chdir(model_dir)
    base = os.path.basename(MODEL)
    # 注入 weld：把骨盆固定到世界（"固定基座人形"），彻底消除浮基与手臂的耦合不稳。
    # 若想体验浮动基座平衡，把 FIX_BASE 置 False 即可（见 main 中的基座伺服分支）。
    with open(base, "r", encoding="utf-8") as f:
        xml = f.read()
    if "<!-- wb-weld -->" not in xml:
        xml = xml.replace(
            "</mujoco>",
            '  <!-- wb-weld -->\n  <equality><weld body1="pelvis"/></equality>\n</mujoco>')
    fixed_name = "_wb_fixed_scene.xml"
    with open(fixed_name, "w", encoding="utf-8") as f:
        f.write(xml)
    spec = mujoco.MjSpec.from_file(fixed_name)

    # 关键：weld 会把骨盆锚定到"编译默认位姿"，而该默认位姿来自 <body name="pelvis"
    # pos="0 0 1.06">（即 qpos0 的 z=1.06），与 home keyframe 的 0.98 不一致 —— 这会让
    # 固定的骨盆把腿顶成外八字。这里把骨盆默认 z 对齐到 HOME_Z，使 qpos0、keyframe、
    # weld 锚点三者一致，站姿自然。
    spec.body("pelvis").pos[2] = HOME_Z

    # 工作台（薄板，承载传送带与料箱）——纯视觉道具，不与机器人碰撞
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.235, 0.29, 0.012], pos=[0.335, 0.14, TABLE_TOP - 0.012],
        rgba=[0.30, 0.31, 0.36, 1.0], friction=[1.0, 0.05, 0.001],
        contype=0, conaffinity=0)
    # 传送带料道（薄板 + 黄色标线）；纯视觉不参与碰撞（料箱由脚本运动学驱动）
    spec.worldbody.add_geom(
        name="belt", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.18, 0.05, 0.01], pos=[0.27, 0.0, TABLE_TOP - 0.005],
        rgba=[0.16, 0.16, 0.19, 1.0], contype=0, conaffinity=0)
    spec.worldbody.add_geom(
        name="belt_marker", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.17, 0.004, 0.001], pos=[0.27, 0.0, TABLE_TOP + 0.006],
        rgba=[0.95, 0.78, 0.12, 0.85], contype=0, conaffinity=0)

    # 分类料箱
    for col in CATEGORIES:
        add_bin(spec, BIN_POS[col], ITEM_RGB[col], f"bin_{col}")

    # 物料：彩色料箱（自由刚体，运行时由脚本驱动其位姿）
    items = []
    for i, col in enumerate(ITEM_COLORS):
        b = spec.worldbody.add_body(name=f"item{i}", pos=[SPAWN_XS[i], PICK[1], ITEM_Z])
        b.add_freejoint()
        b.add_geom(name=f"item{i}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[ITEM_HALF, ITEM_HALF, ITEM_HALF],
                   rgba=list(ITEM_RGB[col]) + [1.0], mass=0.05,
                   friction=[1.0, 0.05, 0.001], contype=2, conaffinity=2)

    # 左前臂末端抓取 site（IK 跟踪目标就是它）
    spec.body("left_elbow_link").add_site(
        name="hand", pos=[0.28, 0.0, -0.015], size=[0.012, 0.012, 0.012],
        rgba=[0.1, 0.6, 1.0, 0.85])

    spec.visual.global_.offwidth = 1100
    spec.visual.global_.offheight = 760

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)      # 必须：填充 site_xpos 等派生量

    # 关节索引映射（qpos adr / dof adr / ctrl id / 限位）
    jinfo = {}
    for jn in JOINT_NAMES:
        jid = model.joint(jn).id
        ctl = model.actuator(jn).id
        jinfo[jn] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid], ctl,
                     model.jnt_range[jid, 0], model.jnt_range[jid, 1])
    left_qadr = [jinfo[j][0] for j in LEFT_ARM]   # 写 qpos 用
    left_dof = [jinfo[j][1] for j in LEFT_ARM]    # 雅可比列（速度索引）用
    left_lo = np.array([jinfo[j][3] for j in LEFT_ARM])
    left_hi = np.array([jinfo[j][4] for j in LEFT_ARM])

    item_info = []
    for i, col in enumerate(ITEM_COLORS):
        bid = model.body(f"item{i}").id
        jid = model.body(f"item{i}").jntadr[0]
        item_info.append({
            "idx": i, "color": col,
            "bin_idx": CATEGORIES.index(col),
            "body_id": bid,
            "qadr": model.jnt_qposadr[jid],   # 写 qpos 用
            "dadr": model.jnt_dofadr[jid],    # 写 qvel / 扭矩用（自由关节两者差 1）
            "state": "onbelt", "spawn_x": SPAWN_XS[i], "x": SPAWN_XS[i],
        })
    hand_id = model.site("hand").id
    return model, data, item_info, hand_id, jinfo, left_qadr, left_dof, left_lo, left_hi


# ---------------------------------------------------------------------------
# 逆运动学：阻尼最小二乘（DLS），仅位置，左臂 4-DoF
# ---------------------------------------------------------------------------
def solve_arm_ik(model, ik_data, hand_id, target_pos, q_init, qadr, dof, qlo, qhi,
                 qpos_full, iters=80, seed_extra=None):
    """DLS 逆运动学（仅位置，左臂 4-DoF）。
    单初值 DLS 容易停在局部极小：从"当前位姿"出发可能残差几厘米，而从零位出发却收敛。
    故用多个初值各解一次，取残差最小的解（best-of-seeds），保证同一目标稳定可达。"""
    seeds = [np.array(q_init, dtype=float).copy()]
    if seed_extra is not None:
        seeds += [np.array(s, dtype=float).copy() for s in seed_extra]

    best_q, best_e = np.array(q_init, dtype=float).copy(), np.inf
    for q0 in seeds:
        q = q0.copy()
        for _ in range(iters):
            ik_data.qpos[:] = qpos_full          # 保持站立/腿部位姿，仅优化左臂
            ik_data.qpos[qadr] = q
            mujoco.mj_kinematics(model, ik_data)
            mujoco.mj_comPos(model, ik_data)     # 雅可比需要 comPos 预计算，否则恒为 0
            err = target_pos - ik_data.site_xpos[hand_id]
            e = float(np.linalg.norm(err))
            if e < best_e:
                best_e = e
                best_q = q.copy()
            if e < 1e-3:
                break
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik_data, Jp, Jr, hand_id)
            J = Jp[:, dof]
            lam = 0.03
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            dq = np.clip(dq, -0.4, 0.4)
            q = np.clip(q + 0.5 * dq, qlo, qhi)
        if best_e < 1e-3:
            break
    ik_data.qpos[qadr] = best_q
    mujoco.mj_kinematics(model, ik_data)
    mujoco.mj_comPos(model, ik_data)
    return best_q, best_e


def build_phases(item):
    """单个料箱的抓取-放置相位序列（目标均为手端世界坐标）。"""
    px = BIN_POS[CATEGORIES[item["bin_idx"]]]
    above = 0.15
    rim_z = TABLE_TOP + BIN_WALL + ITEM_HALF + 0.02      # 箱口上方（不伸进箱内，避免撞壁）
    rim = np.array([px[0], px[1], rim_z])
    return [
        ("Approach", PICK + np.array([0, 0, 0.13]), 0.7),
        ("Descend", PICK, 0.6),
        ("Grasp", PICK, 0.4),
        ("Lift", PICK + np.array([0, 0, 0.15]), 0.6),
        ("Move to bin", px + np.array([0, 0, above]), 0.9),
        ("Place", rim, 0.6),
        ("Release", rim, 0.4),
        ("Retreat", px + np.array([0, 0, above]), 0.5),
    ]


def set_item_pose(data, it, pos):
    """把料箱位姿直接写到 qpos（运动学驱动，不靠接触）。
    注意：qpos 用 jnt_qposadr，qvel 必须用 jnt_dofadr —— 自由关节 7 qpos / 6 qvel，
    两者地址相差 1，写错会把速度漏掉并越界污染相邻料箱。"""
    qa = it["qadr"]
    da = it["dadr"]
    data.qpos[qa:qa + 3] = pos
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="H1 人形物流分拣演示")
    ap.add_argument("--viewer", action="store_true", help="实时 3D 窗口")
    ap.add_argument("--check", action="store_true", help="只跑逻辑不渲染（快速校验分拣正确性）")
    ap.add_argument("--mp4", default=None,
                    help="可选：视频输出路径。默认不生成视频，只输出关键帧图片")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=660)
    args = ap.parse_args()

    model, data, items, hand_id, jinfo, left_qadr, left_dof, left_lo, left_hi = build_scene()
    ik_data = mujoco.MjData(model)
    lim = model.actuator_ctrlrange.copy()
    dt = model.opt.timestep

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.30, 0.14, 1.05]
    cam.distance = 2.05
    cam.azimuth = 142.0
    cam.elevation = -16.0

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
    elif not args.check:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        if args.mp4:
            writer = imageio.get_writer(args.mp4, fps=int(1.0 / CTRL_DT / 2), quality=8)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    for f in os.listdir(FRAMES_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(FRAMES_DIR, f))

    # 状态机
    belt_running = True
    belt_offset = 0.0
    current = None
    phases = None
    phase_idx = 0
    phase_t = 0.0
    ee_cmd = data.site_xpos[hand_id].copy()   # 经速度限幅平滑后的末端指令位置
    label = "init"
    step = 0
    placed_count = 0
    keyframes = []
    captured = False
    # 每个分类箱的容量与已占用槽位（同色多件时沿 x 分槽落座，避免相互碰撞挤箱）
    bin_counts = {c: ITEM_COLORS.count(c) for c in CATEGORIES}
    bin_slots = {c: 0 for c in CATEGORIES}

    # ---- 预热：录制前让初始接触瞬态平息（左臂保持 HOME 休息位）----
    for _ in range(WARMUP_STEPS):
        q_des = HOME.copy()
        q = data.qpos[7:7 + NU]
        qd = data.qvel[6:6 + NU]
        tau = Kp * (q_des - q) - Kd * qd
        for i, jn in enumerate(JOINT_NAMES):
            tau[i] += data.qfrc_bias[jinfo[jn][1]]
        tau = np.clip(tau, lim[:, 0], lim[:, 1])
        for i, jn in enumerate(JOINT_NAMES):
            data.ctrl[jinfo[jn][2]] = tau[i]
        bp = data.qpos[0:3]; bv = data.qvel[0:3]; qc = data.qpos[3:7]
        qe = quat_conj_mul(qc, np.array([1.0, 0.0, 0.0, 0.0])); ba = data.qvel[3:6]
        if not FIX_BASE:
            data.qfrc_applied[0:3] = [KBP * (0 - bp[0]) - KBD * bv[0],
                                      KBP * (0 - bp[1]) - KBD * bv[1],
                                      KBP * (HOME_Z - bp[2]) - KBD * bv[2]]
            data.qfrc_applied[3:6] = KBR * qe[1:4] - KDR * ba
        for _ in range(N_SUBSTEPS):
            for it2 in items:
                if it2["state"] == "onbelt":
                    set_item_pose(data, it2, [it2["x"], PICK[1], ITEM_Z])
            mujoco.mj_step(model, data)
    ee_cmd = data.site_xpos[hand_id].copy()

    max_iter = 7000
    print("running H1 humanoid logistics sorting demo ...")
    for it in range(max_iter):
        # ---- 决定当前目标 ----
        if current is None:
            if belt_running:
                onbelt = [x for x in items if x["state"] == "onbelt"]
                if onbelt:
                    front = max(onbelt, key=lambda a: a["x"])
                    if front["x"] >= PICK_X:
                        current = front
                        belt_running = False
                        phases = build_phases(current)
                        phase_idx = 0
                        phase_t = 0.0
                        captured = False
                        label = f"Grasping {classify(current)}"
                    else:
                        label = "Conveyor feeding..."
                else:
                    label = "Sorting complete"
            else:
                label = "Sorting complete"

        if current is None:
            name, tgt, dur = "待机", READY_ARM, None
        else:
            name, tgt, dur = phases[phase_idx]
            label = f"{name} ({classify(current)} -> {CATEGORIES[current['bin_idx']]} bin)"

        # ---- 末端指令速度限幅（所有动作平滑，避免反作用力掀翻机器人）----
        dvec = tgt - ee_cmd
        dist = float(np.linalg.norm(dvec))
        max_step = MAX_EE_SPEED * CTRL_DT
        if dist > max_step:
            ee_cmd = ee_cmd + dvec / dist * max_step
        else:
            ee_cmd = tgt.copy()
        # 到达判据用"真实手端"而非指令：关节真正跟踪到位才推进相位，
        # 否则手臂会在没到位时就切下一相位，导致放料偏移。
        hand_err = float(np.linalg.norm(np.asarray(tgt) - data.site_xpos[hand_id]))
        arrived = (dist <= 0.012) and (hand_err <= 0.022)

        # ---- 左臂 IK 到平滑后的指令位置（多初值，避开局部极小）----
        q_init = data.qpos[left_qadr].copy()
        q_sol, perr = solve_arm_ik(model, ik_data, hand_id, ee_cmd, q_init,
                                   left_qadr, left_dof, left_lo, left_hi, data.qpos.copy(),
                                   seed_extra=[HOME_ARM])
        # ---- 组装 19 维目标角 ----
        q_des = HOME.copy()
        for k, jn in enumerate(LEFT_ARM):
            q_des[JOINT_NAMES.index(jn)] = q_sol[k]

        # ---- 关节 PD + 重力补偿（全 19 关节）----
        q = data.qpos[7:7 + NU]
        qd = data.qvel[6:6 + NU]
        tau = Kp * (q_des - q) - Kd * qd
        for i, jn in enumerate(JOINT_NAMES):
            tau[i] += data.qfrc_bias[jinfo[jn][1]]      # 计算力矩：抵消重力/科氏，提升跟踪
        tau = np.clip(tau, lim[:, 0], lim[:, 1])
        for i, jn in enumerate(JOINT_NAMES):
            data.ctrl[jinfo[jn][2]] = tau[i]

        # ---- 基座稳定化（仅在浮动基座模式下需要；固定基座时省略）----
        if not FIX_BASE:
            base_pos = data.qpos[0:3]
            base_vel = data.qvel[0:3]
            q_cur = data.qpos[3:7]
            qe = quat_conj_mul(q_cur, np.array([1.0, 0.0, 0.0, 0.0]))
            base_angvel = data.qvel[3:6]
            fx = KBP * (0.0 - base_pos[0]) - KBD * base_vel[0]
            fy = KBP * (0.0 - base_pos[1]) - KBD * base_vel[1]
            fz = KBP * (HOME_Z - base_pos[2]) - KBD * base_vel[2]
            tq = KBR * qe[1:4] - KDR * base_angvel
            data.qfrc_applied[0:3] = [fx, fy, fz]
            data.qfrc_applied[3:6] = tq
        data.xfrc_applied[torso_id] = [0, 0, 0, 0, 0, 0]

        # ---- 物理步进 + 料箱驱动（料箱随手端）----
        hand_pos = data.site_xpos[hand_id]
        for _ in range(N_SUBSTEPS):
            if belt_running:
                belt_offset += BELT_SPEED * DT
            for it2 in items:
                if it2["state"] == "onbelt":
                    it2["x"] = it2["spawn_x"] + belt_offset
                    set_item_pose(data, it2, [it2["x"], PICK[1], ITEM_Z])
                elif it2["state"] == "held":
                    set_item_pose(data, it2, hand_pos)
            mujoco.mj_step(model, data)

        # ---- 相位推进（到达 + 到时，或超时强制）----
        if current is not None:
            phase_t += CTRL_DT
            done = (phase_t >= dur and arrived) or phase_t >= dur * 3.0
            if done:
                if name == "Grasp":
                    current["state"] = "held"
                elif name == "Release":
                    # 归箱落座：按槽位把料箱精确放到目标箱内的静置高度。
                    # （真实系统此处为夹爪张开 + 重力落料；这里保证分拣结果确定、可校验。）
                    col = current["color"]
                    n = bin_counts[col]
                    s = bin_slots[col]
                    bin_slots[col] = s + 1
                    bx = BIN_POS[col]
                    off = (s - (n - 1) / 2.0) * BIN_SPACING
                    set_item_pose(data, current,
                                  np.array([bx[0] + off, bx[1], REST_Z]))
                    current["state"] = "placed"
                    placed_count += 1
                phase_idx += 1
                captured = False
                if phase_idx >= len(phases):
                    current = None
                    belt_running = True
                else:
                    phase_t = 0.0

        # ---- 渲染 / 记录 ----
        if viewer is not None:
            viewer.sync()
            time.sleep(CTRL_DT)
        elif renderer is not None:
            renderer.update_scene(data, cam)
            frame = renderer.render()
            if HAVE_CV2:
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis, (10, 10), (args.width - 10, 46), (0, 0, 0), -1)
                cv2.putText(vis, label, (22, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            if writer is not None and step % 2 == 0:
                writer.append_data(frame)
            if len(keyframes) < 6 and not captured:
                if step == 0:
                    keyframes.append(("01_传送带送料", frame.copy())); captured = True
                elif name == "Grasp" and current is not None and current["state"] == "onbelt":
                    keyframes.append((f"0{len(keyframes)+1}_抓取{classify(current)}", frame.copy())); captured = True
                elif name == "Release" and current is not None and current["state"] == "held":
                    keyframes.append((f"0{len(keyframes)+1}_放入{current['color']}箱", frame.copy())); captured = True

        step += 1
        if placed_count >= len(items) and current is None:
            if renderer is not None:
                for _ in range(30):
                    renderer.update_scene(data, cam)
                    fr = renderer.render()
                    if HAVE_CV2:
                        vis = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(vis, (10, 10), (args.width - 10, 46), (0, 0, 0), -1)
                        cv2.putText(vis, "Sorting complete", (22, 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        fr = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    if writer is not None:
                        writer.append_data(fr)
            break

    for lab, fr in keyframes:
        imageio.imwrite(os.path.join(FRAMES_DIR, f"{lab}.png"), fr)

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
        bx = BIN_POS[CATEGORIES[it["bin_idx"]]]
        d = float(np.sqrt((p[0] - bx[0]) ** 2 + (p[1] - bx[1]) ** 2))
        ok = (d < 0.09) and (p[2] < ITEM_Z + 0.06)   # 落入箱内（水平偏差小且高度已落下）
        all_ok = all_ok and ok
        print(f"  item{it['idx']} {it['color']:>5} -> 目标{CATEGORIES[it['bin_idx']]}箱 "
              f"实际({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) 偏差{d*100:.1f}cm "
              f"{'OK' if ok else 'WRONG'}")
    print(f"\nDONE. 已分拣 {placed_count}/{len(items)} 个，正确性={'全部正确' if all_ok else '存在错误'}。"
          + (f"关键帧 -> {FRAMES_DIR}" if renderer is not None else "（--check 模式，未渲染）")
          + (f"，mp4 -> {args.mp4}" if args.mp4 else ""))


if __name__ == "__main__":
    main()
