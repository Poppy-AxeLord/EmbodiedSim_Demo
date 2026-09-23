"""
Unitree H1 人形机器人 — 进阶运动控制展示 (MuJoCo)
============================================================
一套编排好的人形全身运动控制演示。不再是 hello-world 站立演示，
而是一段编排好的全身运动 routine：

  站立平衡 (baseline)
  双足行走步态  —— 双腿反相正弦摆动 + 重心起伏 + 摆臂协同，相机跟拍
  抗扰动平衡    —— 对躯干施加侧向冲量，平衡控制器（力/力矩 + 四元数姿态误差）把人拉回
  全身协同挥手  —— 双臂上举 + 单臂摆动（上肢与下肢解耦的全身控制）
  原地转向      —— 基座偏航力矩 + 躯干联动，整体 pivot

控制架构（逐层可讲解）：
  * 浮动骨盆用 qfrc_applied 施加 力(位置) + 力矩(姿态)，目标姿态用四元数误差
    表达，避免 rpy 奇异；姿态估计来自模型内置 imu site 与自由关节四元数。
  * 19 个驱动关节用 PD 跟踪目标角。
  * 行走/挥手/转向均由解析轨迹生成，无需学习，可直接解释每一步物理含义。

运行：python humanoid_demo.py              # 离线渲染关键帧（图片）
      python humanoid_demo.py --check      # 只跑逻辑不渲染，快速校验是否摔倒
      python humanoid_demo.py --mp4 out.mp4  # 可选：额外输出视频
依赖：mujoco, numpy, imageio, opencv-python-headless（均已在本环境 site-packages 中）
"""
import os
import argparse
import numpy as np
import mujoco
import imageio.v2 as imageio

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

HERE = os.path.dirname(os.path.abspath(__file__))
# 优先用随仓库打包的模型（桌面/ GitHub 副本开箱即用），否则回退到本机 menagerie 路径
_LOCAL_MODEL = os.path.join(HERE, "unitree_h1_model", "scene.xml")
_MENAGERIE_MODEL = r"D:\embodied_ai\assets\menagerie\models\unitree_h1\scene.xml"
MODEL = os.environ.get(
    "HUMANOID_MODEL",
    _LOCAL_MODEL if os.path.exists(_LOCAL_MODEL) else _MENAGERIE_MODEL)
OUT_DIR = HERE
MP4 = os.path.join(OUT_DIR, "humanoid_demo.mp4")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")

# ---- 关节顺序（freejoint 之后，共 19）----
# 0:L_hip_yaw 1:L_hip_roll 2:L_hip_pitch 3:L_knee 4:L_ankle
# 5:R_hip_yaw 6:R_hip_roll 7:R_hip_pitch 8:R_knee 9:R_ankle
# 10:torso
# 11:L_sh_pitch 12:L_sh_roll 13:L_sh_yaw 14:L_elbow
# 15:R_sh_pitch 16:R_sh_roll 17:R_sh_yaw 18:R_elbow
NU = 19
HOME = np.array([0, 0, -0.4, 0.8, -0.4,
                 0, 0, -0.4, 0.8, -0.4,
                 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0], dtype=float)

# 关节 PD 增益：腿部强、手臂轻
Kp = np.ones(NU) * 150.0
Kd = np.ones(NU) * 15.0
ARM = [11, 12, 13, 14, 15, 16, 17, 18]
Kp[ARM] = 40.0
Kd[ARM] = 4.0

# 基座稳定化增益（简化 WBC / 平衡）
KBP = 3000.0   # 位置
KBD = 600.0    # 速度阻尼
KBR = 1500.0   # 姿态（加强，保证行走时直立、停下后能回正）
KDR = 220.0    # 角速度阻尼

HOME_Z = 0.98  # home 关键帧骨盆高度

# ---- routine 时间线（秒）----
T_STAND = 1.5
T_WALK = 6.0
T_STOP = 7.0
T_PUSH = 7.4
T_RECOVER = 8.6
T_WAVE = 10.6
T_TURN = 12.6
T_END = 14.0
GAIT_T = 1.0   # 一个步态周期时长


def smooth(a, b, t):
    t = float(np.clip(t, 0, 1))
    return a + (b - a) * (t * t * (3 - 2 * t))


def quat_conj_mul(q_cur, q_des):
    """误差四元数 conj(q_cur)*q_des (w,x,y,z)，把当前姿态旋回目标姿态"""
    cw, cx, cy, cz = q_cur
    dw, dx, dy, dz = q_des
    ew = cw*dw + cx*dx + cy*dy + cz*dz
    ex = cw*dx - cx*dw - cy*dz + cz*dy
    ey = cw*dy + cx*dz - cy*dw - cz*dx
    ez = cw*dz - cx*dy + cy*dx - cz*dw
    return np.array([ew, ex, ey, ez])


def yaw_quat(yaw):
    """绕 z 轴旋转 yaw 的四元数 (w,x,y,z)"""
    h = yaw / 2.0
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def euler_to_quat(roll, pitch, yaw):
    """roll=x, pitch=y, yaw=z 欧拉角 -> 四元数 (w,x,y,z)"""
    cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
    sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def control_targets(t, x_target):
    """
    返回当前时刻的控制设定：
      q_des    : 19 维关节目标角
      z_des    : 目标骨盆高度
      vx_des   : 目标前进速度（用于积分 x_target）
      yaw_des  : 目标偏航角
      label    : 视频叠加文字
      push     : (Fx,Fy,Fz) 施加在躯干的外部力，无则 None
    """
    q = HOME.copy()
    z_des = HOME_Z
    vx_des = 0.0
    yaw_des = 0.0
    pitch_des = 0.0
    label = "Stand (balance baseline)"
    push = None

    if t < T_STAND:
        # 站立保持
        pass

    elif t < T_WALK:
        # ---- 双足行走步态 ----
        label = "Bipedal walking gait"
        tt = t - T_STAND
        s = 2 * np.pi * tt / GAIT_T            # 步态相位
        A_hip = 0.30
        A_knee = 0.22
        A_ank = 0.15
        # 左腿
        q[2] = HOME[2] + A_hip * np.sin(s)
        q[3] = HOME[3] + A_knee * np.sin(s)    # 前摆时屈膝抬起
        q[4] = HOME[4] + A_ank * np.sin(s)
        # 右腿（反相）
        q[7] = HOME[7] + A_hip * np.sin(s + np.pi)
        q[8] = HOME[8] + A_knee * np.sin(s + np.pi)
        q[9] = HOME[9] + A_ank * np.sin(s + np.pi)
        # 重心起伏（每步两次）
        z_des = HOME_Z + 0.025 * np.cos(2 * s)
        # 摆臂协同（与同侧腿反相）
        q[11] = -0.25 * np.sin(s)
        q[15] = 0.25 * np.sin(s)
        # 原地行走：不施加前向力，保持完全直立；前向位移由相机后退制造（见 main）
        vx_des = 0.0
        pitch_des = 0.0

    elif t < T_STOP:
        # 收步站稳
        label = "Stop & settle"

    elif t < T_PUSH:
        # ---- 抗扰动平衡：对躯干施加侧向冲量 ----
        label = "Push recovery (disturbance rejection)"
        push = (0.0, 95.0, 0.0)                # +Y 方向 95N 侧向推力

    elif t < T_RECOVER:
        # 受扰后回正站立
        label = "Recover to stand"

    elif t < T_WAVE:
        # ---- 全身协同挥手 ----
        label = "Whole-body wave"
        tt = t - T_RECOVER
        k = smooth(0, 1, min(1.0, tt / 0.4))
        q[3] = HOME[3] + 0.15 * k              # 微屈膝，运动姿态
        q[8] = HOME[8] + 0.15 * k
        q[11] = -1.30 * k                      # 双臂上举
        q[15] = -1.30 * k
        q[14] = 1.00 * k                       # 屈肘
        q[18] = 1.00 * k
        q[17] = 0.70 * np.sin(2 * np.pi * 1.6 * tt) * k   # 右手挥动

    elif t < T_TURN:
        # ---- 原地转向（基座偏航 + 躯干联动）----
        label = "In-place turn (yaw)"
        tt = t - T_WAVE
        span = T_TURN - T_WAVE
        if tt < span / 2:
            yaw_des = 0.40 * smooth(0, 1, tt / (span / 2))
        else:
            yaw_des = 0.40 * smooth(1, 0, (tt - span / 2) / (span / 2))
        q[3] = HOME[3] + 0.10
        q[8] = HOME[8] + 0.10

    else:
        # 回到站立
        label = "Return to stand"

    return q, z_des, vx_des, yaw_des, pitch_des, label, push


def main():
    ap = argparse.ArgumentParser(description="H1 人形进阶运动控制演示")
    ap.add_argument("--check", action="store_true", help="只跑逻辑不渲染（快速校验是否摔倒/失稳）")
    ap.add_argument("--mp4", default=None,
                    help="可选：视频输出路径。默认不生成视频，只输出关键帧图片")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 600
    # 从 home 关键帧初始化
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[kid]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    lim = model.actuator_ctrlrange.copy()
    dt = model.opt.timestep

    renderer = None if args.check else mujoco.Renderer(model, width=900, height=600)
    # 跟拍相机：始终看向骨盆
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 115
    cam.elevation = -12
    cam.distance = 3.8

    os.makedirs(FRAMES_DIR, exist_ok=True)

    FPS = 30
    STEPS_PER_FRAME = 17
    N_FRAMES = int(T_END * FPS)
    frames = []
    z_min, z_max = 1e9, -1e9
    x_target = float(data.qpos[0])
    cam_dist = 3.8             # 行走时相机后拉（dolly out），制造"向前走"观感
    WALK_SPEED_VIS = 0.35      # 视觉前进速度 (m/s)
    print("running advanced humanoid control routine ...")
    for f in range(N_FRAMES):
        t = f / FPS
        q_des, z_des, vx_des, yaw_des, pitch_des, label, push = control_targets(t, x_target)
        for _ in range(STEPS_PER_FRAME):
            x_target += vx_des * dt
            if T_STAND <= t < T_WALK:
                cam_dist += WALK_SPEED_VIS * dt
            else:
                cam_dist += (3.8 - cam_dist) * min(1.0, dt * 2.0)
            # 关节 PD
            q = data.qpos[7:7 + NU]
            qd = data.qvel[6:6 + NU]
            tau = Kp * (q_des - q) - Kd * qd
            tau = np.clip(tau, lim[:, 0], lim[:, 1])
            data.ctrl[:] = tau
            # 基座稳定化（浮动骨盆：力 + 力矩）
            base_pos = data.qpos[0:3]
            base_vel = data.qvel[0:3]
            q_cur = data.qpos[3:7]
            q_ident = euler_to_quat(0.0, pitch_des, yaw_des)
            qe = quat_conj_mul(q_cur, q_ident)
            base_angvel = data.qvel[3:6]
            fx = KBP * (x_target - base_pos[0]) - KBD * base_vel[0]
            fy = KBP * (0.0 - base_pos[1]) - KBD * base_vel[1]
            fz = KBP * (z_des - base_pos[2]) - KBD * base_vel[2]
            tq = KBR * qe[1:4] - KDR * base_angvel
            data.qfrc_applied[0:3] = [fx, fy, fz]
            data.qfrc_applied[3:6] = tq
            # 外部扰动
            if push is not None:
                data.xfrc_applied[torso_id] = [push[0], push[1], push[2], 0, 0, 0]
            else:
                data.xfrc_applied[torso_id] = [0, 0, 0, 0, 0, 0]
            mujoco.mj_step(model, data)
        z = data.qpos[2]
        z_min, z_max = min(z_min, z), max(z_max, z)
        dev = 2 * np.degrees(np.arccos(min(1.0, abs(data.qpos[3]))))
        if f % 30 == 0:
            print(f"frame {f:3d} t {t:5.2f}s base_z {z:.3f} dev {dev:5.1f} deg  {label}")
        # 渲染 + 跟拍（机器人始终居中；行走时相机后拉呈现前进观感）
        if renderer is not None:
            cam.distance = cam_dist
            cam.lookat = np.array([data.qpos[0], data.qpos[1], data.qpos[2]])
            renderer.update_scene(data, cam)
            frame = renderer.render()
            if HAVE_CV2:
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis, (10, 10), (890, 50), (0, 0, 0), -1)
                cv2.putText(vis, label, (24, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
                frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # 关键帧
    if renderer is not None:
        key_t = [0.3, (T_STAND + T_WALK) / 2, T_PUSH - 0.05,
                 (T_RECOVER + T_WAVE) / 2, (T_WAVE + T_TURN) / 2, T_END - 0.1]
        labels = ["01_站立", "02_行走", "03_抗扰动", "04_挥手", "05_转向", "06_收尾"]
        for i, kt in enumerate(key_t):
            ki = int(kt * FPS)
            ki = min(max(ki, 0), len(frames) - 1)
            imageio.imwrite(os.path.join(FRAMES_DIR, f"{labels[i]}.png"), frames[ki])
        renderer.close()

    # 视频为可选产物：默认不出视频，只出关键帧图片
    if args.mp4:
        writer = imageio.get_writer(args.mp4, fps=FPS, macro_block_size=1)
        for fr in frames:
            writer.append_data(fr)
        writer.close()

    fell = z_min < 0.6
    print(f"\nDONE. base_z range: [{z_min:.3f}, {z_max:.3f}] (home={HOME_Z}) "
          f"fell={fell} 渲染帧数={len(frames)}"
          + (f" 关键帧 -> {FRAMES_DIR}" if renderer is not None else "（--check 模式，未渲染）")
          + (f" mp4={args.mp4}" if args.mp4 else ""))


if __name__ == "__main__":
    main()
