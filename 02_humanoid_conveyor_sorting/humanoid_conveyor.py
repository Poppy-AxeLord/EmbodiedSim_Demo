"""
Unitree H1 人形机器人 — 大型物流带分拣演示 (MuJoCo)
=====================================================
**机器人站在整条物流带旁，靠腰部回转身取放**。

与"静止工位、只动手臂"版本的区别：
  * 一条横贯身前的 **大型物流带**（长约 2.5 m）：橡胶带面 + 金属护栏 + 支腿 +
    随带面运动的挡条（可见"带子在跑"）。
  * 料箱**成队列**沿带面送入，前后按 GAP 依次排队，逐件在各自"工位"停下
    （真实物流的 singulation：一件一件来）。
  * 机器人用 **腰部回转（torso yaw，±135° 真关节）+ 左臂** 做**全身协同**：
      - 把左臂摆到物流带左侧/右侧的取件工位（腰角 −67° ~ −21°）
      - 抓取后腰回转到对应颜色出料槽（腰角 −93° ~ +5°）
    单次搬运最大腰部回转 ≈70°，明显区别于"站着只动手"。
  * 三个颜色出料槽（红/蓝/绿）排成弧形，机器人按颜色路由。

控制架构（逐层可讲解）：
  * 固定基座人形：骨盆 weld 到世界（并令锚点与 home 一致，站姿自然）。
  * 19 关节 PD + 重力补偿（qfrc_bias 计算力矩）。
  * **腰角规划**：每个目标点先用「腰+臂」5-DoF 全身 IK 解出一个合适的腰角，
    运行中腰部按**速率限制**平滑回转到该角度；手臂用 4-DoF IK 在"实际腰坐标系"
    里跟踪手端目标 —— 即"腰先转、手动补偿"，稳定且可解释。
  * "感知"层 classify() 独立成函数，换真实视觉模型即可接入，控制层不动。

运行：python humanoid_conveyor.py            # 离线渲染关键帧（图片）
      python humanoid_conveyor.py --viewer   # 实时 3D 窗口
      python humanoid_conveyor.py --check    # 只跑逻辑不渲染，快速校验分拣正确性
      python humanoid_conveyor.py --mp4 out.mp4   # 可选：额外输出视频
依赖：mujoco, numpy, imageio, opencv-python-headless
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
MP4 = os.path.join(OUT_DIR, "humanoid_conveyor.mp4")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")

# ---------------------------------------------------------------------------
# 场景几何（依据「腰+臂」5-DoF 全身 IK 的工作区探针实测设计，残差均 <0.05cm）
# 复现：python probe5.py
# ---------------------------------------------------------------------------
BELT_TOP = 1.01                    # 带面高度
BELT_CX = 0.56                     # 带面中心 x
BELT_HX = 0.09                     # 带面半宽（x 方向）
BELT_Y0, BELT_Y1 = -1.45, 0.85     # 带面纵向范围（长约 2.3m）
BELT_MID = 0.5 * (BELT_Y0 + BELT_Y1)
BELT_HY = 0.5 * (BELT_Y1 - BELT_Y0)

ITEM_HALF = 0.03
ITEM_Z = BELT_TOP + ITEM_HALF      # 料箱在带面上的中心高度 (1.04)
PICK_X = 0.52                      # 取件工位 x（带面近侧）
BELT_SPEED = 0.30                  # 带速 m/s（真实输送带量级）
ITEM_GAP = 0.17                    # 队列中前后料箱的最小间距

# 每件料箱在带面上的"工位"（y 坐标）：大幅错开，迫使腰部来回回转取件
STATION_Y = [-0.42, 0.40, -0.15, 0.15]
# 料箱进入带面时的初始队列位置（index 越大越靠后）
ITEM_INIT_Y = [-0.78, -0.94, -1.10, -1.26]

# 出料槽（红/蓝/绿）排成弧形，位于近侧分拣台上（左右两槽张开到 y=±0.46）
CHUTE = {
    "red":   np.array([0.19, 0.46]),
    "blue":  np.array([0.24, 0.00]),
    "green": np.array([0.19, -0.46]),
}
PLAT_CX = 0.215                    # 分拣台中心 x
# 分拣台面比带面高一点：让料槽整体"高于"带面，视觉上不与带子糊在一起
PLAT_TOP = BELT_TOP + 0.055

BIN_HALF = 0.085
BIN_WALL = 0.07
BIN_SPACING = 0.085                # 同色多件沿 x 分槽落座间距
REST_Z = PLAT_TOP + 0.009 + ITEM_HALF

ITEM_COLORS = ["red", "blue", "green", "red"]
CATEGORIES = ["red", "blue", "green"]
ITEM_RGB = {
    "red":   (0.85, 0.20, 0.18),
    "blue":  (0.15, 0.45, 0.85),
    "green": (0.20, 0.70, 0.30),
}

# 手臂"待机"位（抬高、避开料箱）
READY_ARM = np.array([0.38, 0.10, 1.26])

# ---- 控制频率 ----
CTRL_DT = 0.02
N_SUBSTEPS = 10
DT = CTRL_DT / N_SUBSTEPS
FPS_APPEND = 2                     # 每 2 个控制步写一帧 => 25fps
MAX_EE_SPEED = 0.30                # 末端指令速度上限 (m/s)
MAX_YAW_SPEED = 1.10               # 腰部回转速率上限 (rad/s)
WARMUP_STEPS = 80

# ---- 关节 ----
NU = 19
HOME = np.array([0, 0, -0.4, 0.8, -0.4,
                 0, 0, -0.4, 0.8, -0.4,
                 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0], dtype=float)
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]
LEFT_ARM = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"]
WHOLE = ["torso"] + LEFT_ARM          # 全身协同：腰 + 左臂
TORSO_IDX = JOINT_NAMES.index("torso")
HOME_ARM = np.zeros(len(LEFT_ARM))

# PD 增益：腿强、腰中、臂轻
Kp = np.ones(NU) * 150.0
Kd = np.ones(NU) * 15.0
ARM = [JOINT_NAMES.index(j) for j in LEFT_ARM]
Kp[ARM] = 45.0
Kd[ARM] = 5.0
Kp[TORSO_IDX] = 300.0                 # 腰部（拖动整个上半身）：给足刚度
Kd[TORSO_IDX] = 30.0

FIX_BASE = True
KBP, KBD, KBR, KDR = 40000.0, 4000.0, 40000.0, 4000.0
HOME_Z = 0.98


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def quat_conj_mul(q_cur, q_des):
    cw, cx, cy, cz = q_cur
    dw, dx, dy, dz = q_des
    ew = cw * dw + cx * dx + cy * dy + cz * dz
    ex = cw * dx - cx * dw - cy * dz + cz * dy
    ey = cw * dy + cx * dz - cy * dw - cz * dx
    ez = cw * dz - cx * dy + cy * dx - cz * dw
    return np.array([ew, ex, ey, ez])


def classify(item):
    """感知模块：返回料箱颜色（真值占位；真实系统为 RGB-D 检测/分割/分类）。"""
    return item["color"]


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------
def add_chute(spec, pos, rgb, name):
    """出料槽：开口朝上的浅箱，顶部有一圈亮色沿口（便于肉眼识别"槽"）。
    只与料箱(物品)互碰，不与机器人碰撞。"""
    b = spec.worldbody.add_body(name=name, pos=[pos[0], pos[1], PLAT_TOP])
    # 底
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, BIN_HALF, 0.006],
               pos=[0, 0, 0.003], rgba=[0.18, 0.18, 0.21, 1.0], friction=[1, 0.05, 0.001],
               contype=2, conaffinity=2)
    cz = BIN_WALL / 2 + 0.006
    rim_z = cz + BIN_WALL / 2
    for dx in (BIN_HALF, -BIN_HALF):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, BIN_HALF, BIN_WALL / 2],
                   pos=[dx, 0, cz], rgba=list(rgb) + [0.75], contype=2, conaffinity=2)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.007, BIN_HALF, 0.005],
                   pos=[dx, 0, rim_z], rgba=list(rgb) + [1.0], contype=2, conaffinity=2)
    for dy in (BIN_HALF, -BIN_HALF):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, 0.005, BIN_WALL / 2],
                   pos=[0, dy, cz], rgba=list(rgb) + [0.75], contype=2, conaffinity=2)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, 0.007, 0.005],
                   pos=[0, dy, rim_z], rgba=list(rgb) + [1.0], contype=2, conaffinity=2)


CLEAT_NAMES = []


def build_belt(spec):
    """大型物流带：带面 + 金属框架 + 护栏 + 支腿 + 运动挡条（纯视觉，不参与碰撞）。"""
    # 带面（橡胶，深色）
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[BELT_HX, BELT_HY, 0.012],
        pos=[BELT_CX, BELT_MID, BELT_TOP - 0.012],
        rgba=[0.13, 0.13, 0.15, 1.0], contype=0, conaffinity=0)
    # 金属框架（带面下方）
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[BELT_HX + 0.006, BELT_HY, 0.05],
        pos=[BELT_CX, BELT_MID, BELT_TOP - 0.075],
        rgba=[0.55, 0.57, 0.60, 1.0], contype=0, conaffinity=0)
    # 两侧护栏
    for dx in (BELT_HX + 0.006, -(BELT_HX + 0.006)):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, BELT_HY, 0.030],
            pos=[BELT_CX + dx, BELT_MID, BELT_TOP + 0.018],
            rgba=[0.72, 0.74, 0.78, 1.0], contype=0, conaffinity=0)
    # 支腿
    for dx in (BELT_HX - 0.02, -(BELT_HX - 0.02)):
        for y in (BELT_Y0 + 0.25, BELT_Y1 - 0.25):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.014, 0.014, (BELT_TOP - 0.10) / 2],
                pos=[BELT_CX + dx, y, (BELT_TOP - 0.10) / 2],
                rgba=[0.45, 0.47, 0.50, 1.0], contype=0, conaffinity=0)
    # 运动挡条（随带面移动，可见"带子在跑"）
    n = 14
    span = BELT_Y1 - BELT_Y0
    for k in range(n):
        y0 = BELT_Y0 + k * span / n
        b = spec.worldbody.add_body(name=f"cleat{k}", pos=[BELT_CX, y0, BELT_TOP + 0.004])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BELT_HX - 0.005, 0.010, 0.005],
                   rgba=[0.90, 0.72, 0.14, 0.95], contype=0, conaffinity=0)
        CLEAT_NAMES.append(f"cleat{k}")


def build_platform(spec):
    """近侧分拣台（承载出料槽）：台面 + 立柜。纯视觉，不与机器人碰撞。"""
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.115, 0.60, 0.012],
        pos=[PLAT_CX, 0.0, PLAT_TOP - 0.012],
        rgba=[0.32, 0.33, 0.38, 1.0], contype=0, conaffinity=0)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.10, 0.57, (PLAT_TOP - 0.024) / 2],
        pos=[PLAT_CX, 0.0, (PLAT_TOP - 0.024) / 2],
        rgba=[0.24, 0.25, 0.29, 1.0], contype=0, conaffinity=0)
    # 台面上按料槽位置铺一条同色标线，强化"这是分拣位不是带子"的读图
    for col in CATEGORIES:
        ch = CHUTE[col]
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.115, BIN_HALF + 0.012, 0.002],
            pos=[PLAT_CX, ch[1], PLAT_TOP + 0.003],
            rgba=list(ITEM_RGB[col]) + [0.30], contype=0, conaffinity=0)


def build_scene():
    model_dir = os.path.dirname(MODEL)
    os.chdir(model_dir)
    base = os.path.basename(MODEL)
    # 注入 weld：骨盆固定到世界（"固定基座人形"），消除浮基与手臂的耦合不稳。
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

    # weld 锚点用"编译默认位姿"= <body name="pelvis" pos="0 0 1.06"> 的 z=1.06，
    # 与 home keyframe 的 0.98 不一致，会把腿顶成外八字。对齐到 HOME_Z 后站姿自然。
    spec.body("pelvis").pos[2] = HOME_Z

    build_belt(spec)
    build_platform(spec)
    for col in CATEGORIES:
        add_chute(spec, CHUTE[col], ITEM_RGB[col], f"chute_{col}")

    # 料箱（自由刚体，运行时由脚本驱动位姿）
    for i, col in enumerate(ITEM_COLORS):
        b = spec.worldbody.add_body(name=f"item{i}", pos=[PICK_X, ITEM_INIT_Y[i], ITEM_Z])
        b.add_freejoint()
        b.add_geom(name=f"item{i}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[ITEM_HALF, ITEM_HALF, ITEM_HALF],
                   rgba=list(ITEM_RGB[col]) + [1.0], mass=0.05,
                   friction=[1.0, 0.05, 0.001], contype=2, conaffinity=2)

    # 左前臂末端抓取 site
    spec.body("left_elbow_link").add_site(
        name="hand", pos=[0.28, 0.0, -0.015], size=[0.012, 0.012, 0.012],
        rgba=[0.1, 0.6, 1.0, 0.85])

    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    jinfo = {}
    for jn in JOINT_NAMES:
        jid = model.joint(jn).id
        jinfo[jn] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid],
                     model.actuator(jn).id, model.jnt_range[jid, 0], model.jnt_range[jid, 1])
    left_qadr = [jinfo[j][0] for j in LEFT_ARM]
    left_dof = [jinfo[j][1] for j in LEFT_ARM]
    left_lo = np.array([jinfo[j][3] for j in LEFT_ARM])
    left_hi = np.array([jinfo[j][4] for j in LEFT_ARM])

    whole_qadr = np.array([jinfo[j][0] for j in WHOLE])
    whole_dof = np.array([jinfo[j][1] for j in WHOLE])
    whole_lo = np.array([jinfo[j][3] for j in WHOLE])
    whole_hi = np.array([jinfo[j][4] for j in WHOLE])

    item_info = []
    for i, col in enumerate(ITEM_COLORS):
        jid = model.body(f"item{i}").jntadr[0]
        item_info.append({
            "idx": i, "color": col,
            "bin_idx": CATEGORIES.index(col),
            "qadr": model.jnt_qposadr[jid],     # 写 qpos 用
            "dadr": model.jnt_dofadr[jid],      # 写 qvel 用（自由关节两者差 1）
            "state": "onbelt", "y": ITEM_INIT_Y[i], "station_y": STATION_Y[i],
        })
    hand_id = model.site("hand").id
    cleat_ids = [model.body(nm).id for nm in CLEAT_NAMES]
    cleat_base = [ITEM_INIT_Y[0] for _ in CLEAT_NAMES]  # 占位，实际用初始 y
    cleat_base = [float(model.body_pos[bid][1]) for bid in cleat_ids]
    return (model, data, item_info, hand_id, jinfo, left_qadr, left_dof, left_lo, left_hi,
            whole_qadr, whole_dof, whole_lo, whole_hi, cleat_ids, cleat_base)


# ---------------------------------------------------------------------------
# 逆运动学
# ---------------------------------------------------------------------------
def _dls(model, idata, hand_id, target, q, qadr, dof, qlo, qhi, qpos_full,
         iters, step, lam):
    best_q, best_e = q.copy(), np.inf
    for _ in range(iters):
        idata.qpos[:] = qpos_full
        idata.qpos[qadr] = q
        mujoco.mj_kinematics(model, idata)
        mujoco.mj_comPos(model, idata)          # 雅可比需要 comPos，否则恒为 0
        err = target - idata.site_xpos[hand_id]
        e = float(np.linalg.norm(err))
        if e < best_e:
            best_e, best_q = e, q.copy()
        if e < 5e-4:
            break
        Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, idata, Jp, Jr, hand_id)
        J = Jp[:, dof]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        dq = np.clip(dq, -0.4, 0.4)
        q = np.clip(q + step * dq, qlo, qhi)
    return best_q, best_e


def solve_arm_ik(model, idata, hand_id, target, q_init, qadr, dof, qlo, qhi,
                 qpos_full, iters=80, seed_extra=None):
    """左臂 4-DoF DLS IK（固定腰坐标系下）。多初值 best-of-seeds 避免局部极小。"""
    seeds = [np.array(q_init, float).copy()]
    if seed_extra is not None:
        seeds += [np.array(s, float).copy() for s in seed_extra]
    best_q, best_e = np.array(q_init, float).copy(), np.inf
    for q0 in seeds:
        q, e = _dls(model, idata, hand_id, target, q0, qadr, dof, qlo, qhi,
                    qpos_full, iters, 0.5, 0.03)
        if e < best_e:
            best_e, best_q = e, q
        if best_e < 5e-4:
            break
    return best_q, best_e


def solve_whole_ik(model, idata, hand_id, target, q_init, qadr, dof, qlo, qhi,
                   qpos_full, iters=140):
    """「腰 + 左臂」5-DoF 全身 IK：用于**规划**每个目标点合适的腰角。
    3 维位置目标 + 5 个自由度 => 2 个冗余自由度，DLS 自然分配（腰部回转帮忙够远侧）。"""
    q, e = _dls(model, idata, hand_id, target, np.array(q_init, float), qadr, dof,
                qlo, qhi, qpos_full, iters, 0.6, 0.05)
    return q, e


def plan_yaw(model, idata, hand_id, target, qpos_ref, whole_qadr, whole_dof,
             whole_lo, whole_hi):
    """给定手端目标，用 5-DoF 全身 IK 解出建议腰角（rad）。"""
    q, e = solve_whole_ik(model, idata, hand_id, np.asarray(target, float),
                          np.zeros(5), whole_qadr, whole_dof, whole_lo, whole_hi, qpos_ref)
    return float(q[0]), float(e)


def build_phases(item):
    """单件料箱的取-放相位（目标均为手端世界坐标）。"""
    sy = item["station_y"]
    pick = np.array([PICK_X, sy, ITEM_Z])
    ch = CHUTE[item["color"]]
    rim_z = PLAT_TOP + BIN_WALL + ITEM_HALF + 0.02
    rim = np.array([ch[0], ch[1], rim_z])
    return [
        ("Approach", pick + np.array([0, 0, 0.14]), 0.7),
        ("Descend", pick, 0.6),
        ("Grasp", pick, 0.4),
        ("Lift", pick + np.array([0, 0, 0.16]), 0.6),
        ("Turn to chute", rim + np.array([0, 0, 0.10]), 0.9),
        ("Place", rim, 0.6),
        ("Release", rim, 0.4),
        ("Retreat", rim + np.array([0, 0, 0.16]), 0.6),
    ]


# ---------------------------------------------------------------------------
# 料箱驱动
# ---------------------------------------------------------------------------
def set_item_pose(data, it, pos):
    """运动学驱动料箱位姿。qpos 用 jnt_qposadr，qvel 必须用 jnt_dofadr
    （自由关节 7 qpos / 6 qvel，混用会漏速度并越界污染相邻料箱）。"""
    qa, da = it["qadr"], it["dadr"]
    data.qpos[qa:qa + 3] = pos
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0


def advance_belt(items, dt):
    """输送带队列：所有在带料箱以带速前进，前车（y 最大）停在自己的工位，
    后车被前车挡住（间距 >= ITEM_GAP）—— 真实物流的 singulation。"""
    belt = sorted([x for x in items if x["state"] == "onbelt"],
                  key=lambda a: -a["y"])
    limit = None
    for it in belt:
        new_y = it["y"] + BELT_SPEED * dt
        if limit is not None:
            new_y = min(new_y, limit)
        new_y = min(new_y, it["station_y"])
        it["y"] = new_y
        limit = it["y"] - ITEM_GAP


def update_cleats(model, cleat_ids, cleat_base, offset):
    span = BELT_Y1 - BELT_Y0
    for bid, y0 in zip(cleat_ids, cleat_base):
        y = BELT_Y0 + ((y0 + offset - BELT_Y0) % span)
        model.body_pos[bid][1] = y


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="H1 大型物流带分拣演示")
    ap.add_argument("--viewer", action="store_true", help="实时 3D 窗口")
    ap.add_argument("--check", action="store_true", help="只跑逻辑不渲染（快速校验分拣正确性）")
    ap.add_argument("--mp4", default=None,
                    help="可选：视频输出路径。默认不生成视频，只输出关键帧图片")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    (model, data, items, hand_id, jinfo, left_qadr, left_dof, left_lo, left_hi,
     w_qadr, w_dof, w_lo, w_hi, cleat_ids, cleat_base) = build_scene()
    ik_data = mujoco.MjData(model)
    lim = model.actuator_ctrlrange.copy()
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.30, -0.22, 1.06]
    cam.distance = 2.95
    cam.azimuth = 212.0
    cam.elevation = -28.0

    renderer = viewer = writer = None
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
            writer = imageio.get_writer(args.mp4, fps=int(1.0 / CTRL_DT / FPS_APPEND), quality=8)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    if renderer is not None:
        for f in os.listdir(FRAMES_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(FRAMES_DIR, f))

    # ---- 预热：平息初始接触瞬态 ----
    for _ in range(WARMUP_STEPS):
        q = data.qpos[7:7 + NU]; qd = data.qvel[6:6 + NU]
        tau = Kp * (HOME - q) - Kd * qd
        for i, jn in enumerate(JOINT_NAMES):
            tau[i] += data.qfrc_bias[jinfo[jn][1]]
        tau = np.clip(tau, lim[:, 0], lim[:, 1])
        for i, jn in enumerate(JOINT_NAMES):
            data.ctrl[jinfo[jn][2]] = tau[i]
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)
    qpos_ref = data.qpos.copy()          # 稳定站姿（腰角规划的参考构型）

    # ---- 状态机 ----
    belt_offset = 0.0
    current = None
    phases = None
    phase_idx = 0
    phase_t = 0.0
    ee_cmd = data.site_xpos[hand_id].copy()
    torso_cmd = 0.0
    yaw_cache = {}
    label = "init"
    step = 0
    placed_count = 0
    yaw_abs_max = 0.0
    keyframes = []
    captured = set()
    bin_slots = {c: 0 for c in CATEGORIES}
    bin_counts = {c: ITEM_COLORS.count(c) for c in CATEGORIES}

    def yaw_for(target):
        key = tuple(np.round(np.asarray(target, float), 3))
        if key not in yaw_cache:
            y, e = plan_yaw(model, ik_data, hand_id, np.asarray(target, float),
                            qpos_ref, w_qadr, w_dof, w_lo, w_hi)
            yaw_cache[key] = y
        return yaw_cache[key]

    max_iter = 14000
    print("running H1 humanoid CONVEYOR sorting demo ...")
    for it in range(max_iter):
        # ---- 取件调度：带面队首到工位 -> 开始抓取 ----
        if current is None:
            onbelt = [x for x in items if x["state"] == "onbelt"]
            if onbelt:
                front = max(onbelt, key=lambda a: a["y"])
                if abs(front["y"] - front["station_y"]) < 1e-6:
                    current = front
                    phases = build_phases(current)
                    phase_idx = 0
                    phase_t = 0.0
                    label = f"Pick {classify(current)}"
                else:
                    label = "Conveyor feeding..."
            else:
                label = "Sorting complete"

        if current is None:
            name, tgt, dur = "待机", READY_ARM, None
            yaw_target = 0.0
        else:
            name, tgt, dur = phases[phase_idx]
            yaw_target = yaw_for(tgt)
            label = f"{name} ({classify(current)} -> {CATEGORIES[current['bin_idx']]})"

        # ---- 腰部：按速率限制平滑回转到规划腰角 ----
        dyaw = np.clip(yaw_target - torso_cmd, -MAX_YAW_SPEED * CTRL_DT,
                       MAX_YAW_SPEED * CTRL_DT)
        torso_cmd += dyaw

        # ---- 末端指令速度限幅 ----
        dvec = np.asarray(tgt, float) - ee_cmd
        dist = float(np.linalg.norm(dvec))
        max_step = MAX_EE_SPEED * CTRL_DT
        if dist > max_step:
            ee_cmd = ee_cmd + dvec / dist * max_step
        else:
            ee_cmd = np.asarray(tgt, float).copy()
        hand_err = float(np.linalg.norm(np.asarray(tgt, float) - data.site_xpos[hand_id]))
        arrived = (dist <= 0.014) and (hand_err <= 0.024)

        # ---- 左臂 IK（以"实际腰"为参考构型，腰转到位手臂自动补偿）----
        q_init = data.qpos[left_qadr].copy()
        q_sol, _ = solve_arm_ik(model, ik_data, hand_id, ee_cmd, q_init,
                                left_qadr, left_dof, left_lo, left_hi, data.qpos.copy(),
                                seed_extra=[HOME_ARM])

        # ---- 组装 19 维目标角（腰 = 规划值，臂 = IK 解）----
        q_des = HOME.copy()
        q_des[TORSO_IDX] = torso_cmd
        for k, jn in enumerate(LEFT_ARM):
            q_des[JOINT_NAMES.index(jn)] = q_sol[k]

        # ---- 关节 PD + 重力补偿 ----
        q = data.qpos[7:7 + NU]; qd = data.qvel[6:6 + NU]
        tau = Kp * (q_des - q) - Kd * qd
        for i, jn in enumerate(JOINT_NAMES):
            tau[i] += data.qfrc_bias[jinfo[jn][1]]
        tau = np.clip(tau, lim[:, 0], lim[:, 1])
        for i, jn in enumerate(JOINT_NAMES):
            data.ctrl[jinfo[jn][2]] = tau[i]
        data.xfrc_applied[torso_id] = [0, 0, 0, 0, 0, 0]

        # ---- 物理步进 + 料箱驱动（带面队列 / 随手端）----
        hand_pos = data.site_xpos[hand_id]
        for _ in range(N_SUBSTEPS):
            belt_offset += BELT_SPEED * DT
            advance_belt(items, DT)
            for it2 in items:
                if it2["state"] == "onbelt":
                    set_item_pose(data, it2, [PICK_X, it2["y"], ITEM_Z])
                elif it2["state"] == "held":
                    set_item_pose(data, it2, hand_pos)
            mujoco.mj_step(model, data)
        update_cleats(model, cleat_ids, cleat_base, belt_offset)
        yaw_abs_max = max(yaw_abs_max, abs(float(data.qpos[jinfo["torso"][0]])))

        # ---- 相位推进 ----
        if current is not None:
            phase_t += CTRL_DT
            done = (phase_t >= dur and arrived) or phase_t >= dur * 3.0
            if done:
                if name == "Grasp":
                    current["state"] = "held"
                elif name == "Release":
                    col = current["color"]
                    s = bin_slots[col]; bin_slots[col] = s + 1
                    n = bin_counts[col]
                    off = (s - (n - 1) / 2.0) * BIN_SPACING
                    ch = CHUTE[col]
                    set_item_pose(data, current, np.array([ch[0] + off, ch[1], REST_Z]))
                    current["state"] = "placed"
                    placed_count += 1
                phase_idx += 1
                if phase_idx >= len(phases):
                    current = None
                else:
                    phase_t = 0.0

        # ---- 渲染 / 关键帧 ----
        if viewer is not None:
            viewer.sync()
            time.sleep(CTRL_DT)
        elif renderer is not None:
            renderer.update_scene(data, cam)
            frame = renderer.render()
            if HAVE_CV2:
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis, (10, 10), (args.width - 10, 52), (0, 0, 0), -1)
                cv2.putText(vis, label, (22, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (0, 255, 255), 2)
                yaw_deg = np.degrees(data.qpos[jinfo["torso"][0]])
                cv2.putText(vis, f"waist yaw {yaw_deg:+6.1f} deg", (22, args.height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
                frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            def cap(key, name_):
                if key not in captured:
                    keyframes.append((name_, frame.copy()))
                    captured.add(key)

            if step == 0:
                cap("feed", "物流带送料")
            if current is not None and name == "Descend" \
                    and abs(data.qpos[jinfo["torso"][0]]) > 0.62:
                cap(f"waist{current['idx']}", f"腰部回转_{classify(current)}")
            if current is not None and name == "Grasp" and current["state"] == "onbelt":
                cap(f"grasp{current['idx']}", f"抓取{classify(current)}")
            if current is not None and name == "Release" and current["state"] == "held":
                cap(f"place{current['idx']}", f"放入{current['color']}槽")

            if writer is not None and step % FPS_APPEND == 0:
                writer.append_data(frame)

        step += 1
        if placed_count >= len(items) and current is None:
            if renderer is None:
                break
            for _ in range(30):
                renderer.update_scene(data, cam)
                fr = renderer.render()
                if HAVE_CV2:
                    vis = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(vis, (10, 10), (args.width - 10, 52), (0, 0, 0), -1)
                    cv2.putText(vis, "Sorting complete", (22, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                    fr = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                if writer is not None:
                    writer.append_data(fr)
            break

    # 关键帧写入：按"送料 -> 大回转取红 -> 抓红 -> 入红槽 -> 回转取绿 -> 入绿槽"叙事排序
    have = [k for k, _ in keyframes]
    pref = ["物流带送料", "腰部回转_red", "放入red槽",
            "腰部回转_green", "抓取green", "放入green槽",
            "腰部回转_blue", "抓取red", "抓取blue", "放入blue槽"]
    order = [lab for lab in pref if lab in have]
    order += [lab for lab in have if lab not in order]
    sel = []
    for lab in order:
        for k, fr in keyframes:
            if k == lab:
                sel.append((fr, lab))
                break
    sel = sel[:6]
    for i, (fr, lab) in enumerate(sel):
        imageio.imwrite(os.path.join(FRAMES_DIR, f"{i+1:02d}_{lab}.png"), fr)

    if writer is not None:
        writer.close(); renderer.close()
    if viewer is not None:
        viewer.close()

    # ---- 校验 ----
    print("\n分拣结果校验：")
    all_ok = True
    for it in items:
        p = data.body(f"item{it['idx']}").xpos
        ch = CHUTE[it["color"]]
        dx, dy = abs(p[0] - ch[0]), abs(p[1] - ch[1])
        d = float(np.hypot(dx, dy))
        ok = (dx < BIN_HALF) and (dy < BIN_HALF) and (abs(p[2] - REST_Z) < 0.05)
        all_ok = all_ok and ok
        print(f"  item{it['idx']} {it['color']:>5} -> {it['color']}槽 "
              f"实际({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) 偏差{d*100:4.1f}cm "
              f"{'OK' if ok else 'WRONG'}")
    print(f"\n腰部回转：全程最大偏航 {np.degrees(yaw_abs_max):.1f} deg "
          f"(关节范围 ±135 deg)")
    print(f"\nDONE. 已分拣 {placed_count}/{len(items)}，正确性={'全部正确' if all_ok else '存在错误'}。"
          f"\n关键帧 {len(sel)} 张 -> {FRAMES_DIR}" + (f"\nmp4 -> {args.mp4}" if args.mp4 else ""))


if __name__ == "__main__":
    main()
