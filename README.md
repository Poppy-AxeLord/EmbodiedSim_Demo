# EmbodiedSim_Demo · 具身智能仿真合集（MuJoCo）

五个自包含的 MuJoCo 机器人仿真项目，覆盖具身智能从**操作**到**运动控制**的主要层次：
机械臂抓取分拣、人形操作（工位 / 全身协同）、灵巧手多指掌内操纵、人形双足运动控制。

每个项目都满足：**纯 CPU 可跑 · 模型随仓库打包 · 一条命令复现 · 结果自动校验**。

---

## 项目一览

| # | 项目 | 机器人（自由度） | 一句话看点 |
|---|---|---|---|
| 01 | [机械臂 · 物流分拣](#01-机械臂--物流分拣) | Franka Emika Panda（7） | 传送带送料 + DLS 逆运动学抓取 + 按色分拣入箱 |
| 02 | [人形 · 物流带分拣](#02-人形--物流带分拣腰部回转--全身协同) | Unitree H1（19） | 2.3 m 输送带 + **腰部回转身取件**，5-DoF 全身 IK |
| 03 | [灵巧手 · 掌内操纵](#03-灵巧手--掌内操纵) | Shadow Hand E3M5（24） | 五指包络抓取 + **四指行波搓球** + 翻腕卸料 |
| 04 | [人形 · 工位分拣](#04-人形--工位分拣) | Unitree H1（19） | 实测站姿标定 + 4-DoF 臂 IK + best-of-seeds 破局部极小 |
| 05 | [人形 · 运动控制](#05-人形--运动控制) | Unitree H1（19） | 双足行走 + **抗扰动平衡** + 全身协同 + 原地转向 |

---

## 01 机械臂 · 物流分拣

Panda 机械臂 + 传送带 + 三色分类箱，完整跑通「感知 → 抓取 → 放置」闭环。

- **模型**：Franka Emika Panda（7 自由度，MuJoCo Menagerie 官方模型）
- **依赖**：`mujoco` `numpy` `imageio`

### 关键帧

| | |
|---|---|
| ![传送带送料](01_arm_logistics_sorting/frames/01_传送带送料.png) **① 传送带送料** — 彩色料箱排队流动到抓取点 | ![抓取 red](01_arm_logistics_sorting/frames/02_抓取red.png) **② 抓取 red** — DLS 逆运动学定位 + 力控夹爪 |
| ![放入 red 箱](01_arm_logistics_sorting/frames/03_放入red箱.png) **③ 放入 red 箱** | ![抓取 blue](01_arm_logistics_sorting/frames/04_抓取blue.png) **④ 抓取 blue** |
| ![放入 blue 箱](01_arm_logistics_sorting/frames/05_放入blue箱.png) **⑤ 放入 blue 箱** | ![抓取 green](01_arm_logistics_sorting/frames/06_抓取green.png) **⑥ 抓取 green** |

### 实测指标

| 指标 | 数值 |
|---|---|
| 分拣正确性 | **4 / 4 全部正确** |
| 入箱定位偏差 | 约 **2 cm** |
| 夹持力 | 约 **12 N**（力控弹簧夹爪） |

### 技术要点

| 技术点 | 实现 |
|---|---|
| 物理仿真 | MuJoCo + Menagerie 官方 **Franka Emika Panda** 7-DoF 模型 |
| 逆运动学 | **阻尼最小二乘（DLS）**：`dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`，雅可比由 `mj_jacSite` 从模型直接获取，不写手工运动学 |
| 关节控制 | 位置执行器跟踪 IK 解；夹爪为力控弹簧执行器 |
| 轨迹平滑 | `smoothstep` 插值，阶段间目标点平滑过渡，避免速度突变 |
| 感知接口 | `classify()` 独立成函数 —— 换成真实视觉模型输出即可接入，控制层无需改动 |
| 场景 | 工作台 + 传送带（带黄色标线）+ 三个彩色分类料箱，全部由 `MjSpec` 程序化构建 |
| 节拍控制 | 抓取期间传送带暂停（stop-and-go 分拣），放完自动续送下一件 |

两个值得展开的工程设计：

- **IK 与仿真状态严格隔离**：逆运动学在独立的 `ik_data` 上做正运动学，绝不修改仿真 `data` ——
  否则每个控制周期把机械臂"瞬移"一次会破坏接触/摩擦，抓取必失败。
- **阻尼抑制奇异位形**：`λ` 项抑制奇异附近的巨大关节速度，并对单步关节增量做限幅，
  避免机械臂在腕部奇异点附近抖振。

### 运行

```bash
python 01_arm_logistics_sorting/logistics_sorting.py            # 出关键帧图片
python 01_arm_logistics_sorting/logistics_sorting.py --viewer   # 实时 3D 窗口
```

---

## 02 人形 · 物流带分拣（腰部回转 / 全身协同）

H1 站在整条物流带旁，靠**腰部回转身**取件、再回转身按颜色分拣入槽。

- **模型**：Unitree H1（19 DoF，MuJoCo Menagerie 官方模型）
- **场景**：自建大型输送带产线（2.3 m 带面 + 护栏 + 支腿 + 运动挡条）

动作链：输送带上料 → 队列进位（singulation）→ 腰部回转取件 → 按色入槽

### 关键帧

| | |
|---|---|
| ![物流带送料](02_humanoid_conveyor_sorting/frames/01_物流带送料.png) **① 物流带送料** — 料箱在带面上排成队列 | ![腰部回转取红](02_humanoid_conveyor_sorting/frames/02_腰部回转_red.png) **② 腰部回转取件** — 腰转到带子另一侧够到红箱 |
| ![放入红槽](02_humanoid_conveyor_sorting/frames/03_放入red槽.png) **③ 放入红槽** — 回转身把红箱送进红槽 | ![腰部回转取绿](02_humanoid_conveyor_sorting/frames/04_腰部回转_green.png) **④ 腰部回转取绿** — 再回转到另一工位 |
| ![抓取绿箱](02_humanoid_conveyor_sorting/frames/05_抓取green.png) **⑤ 抓取绿箱** | ![放入绿槽](02_humanoid_conveyor_sorting/frames/06_放入green槽.png) **⑥ 放入绿槽** |

### 实测指标

| 指标 | 数值 |
|---|---|
| 分拣正确性 | **4 / 4 全部正确** |
| 入槽水平偏差 | ≤ **4.3 cm**（两个同色件同槽分位，属设计值） |
| **腰部回转最大偏航** | **101.2°**（`torso` 关节量程 ±135°） |
| 末端跟踪残差 | ≤ 约 **2 cm** |

### 技术要点

**1. 腰部回转并入逆运动学 —— 5-DoF「全身 IK」**

H1 的 `torso` 是绕 **z 轴**的偏航关节（`axis="0 0 1"`，量程 **±2.35 rad = ±135°**），电机力矩 ±200 N·m。
本项目把 **腰 1 + 左臂 4 = 5 个自由度**一起放进阻尼最小二乘 IK：

```
[Jp[:, dof]] · dq = err          # 3 维位置误差，5 个未知量 => 2 个冗余自由度
dq = Jᵀ (J Jᵀ + λI)⁻¹ err        # λ 抑制奇异
```

腰部回转的物理作用：左肩不在身体中轴上（在 y = +0.155 m），**腰部负向偏航会把左臂甩到身体右侧**，
于是带子另一侧的工位也能被左臂够到。实测手臂可达 y ∈ [−0.45, +0.45]（x = 0.50 处），残差均 < 0.05 cm。

**2. 腰先转、臂补偿（双速率协同）**

```
腰：torso_cmd 以 ≤1.10 rad/s 速率限幅平滑回转到规划角   —— 宏观大动作
臂：4-DoF IK 以「实际腰角」为参考构型跟踪手端目标        —— 微观补偿
```

即"腰负责够远侧、臂负责精定位"。好处：腰的运动连续可解释，手臂精度不受腰部动态影响，
也不会出现 IK 在冗余自由度上乱跳。

**3. 输送带队列（拟真 singulation）**

```python
belt = sorted(在带料箱, key=-y)          # 队首 = y 最大
limit = None
for it in belt:
    new_y = it.y + BELT_SPEED * dt
    if limit: new_y = min(new_y, limit)  # 不许越过前车
    new_y = min(new_y, it.station_y)     # 到自己的工位就停
    it.y, limit = new_y, new_y - ITEM_GAP
```

**4. 站立与驱动**

- **固定基座人形**：骨盆 weld 到世界，并把 weld 锚点与 `home` 关键帧对齐（`HOME_Z = 0.98`），
  避免模型内置 `<body pos="0 0 1.06">` 把腿顶成外八字。
- **计算力矩**：19 关节 PD + `data.qfrc_bias` 重力/科氏补偿；腰部刚度单独加大。
- ⚠️ 自由关节 **7 个 qpos / 6 个 qvel**：写 `qpos` 用 `jnt_qposadr`，写 `qvel` 必须用 `jnt_dofadr`，
  两者混用会漏速度并越界污染相邻料箱。
- 台面/带面/护栏均为**纯视觉道具**（`contype=conaffinity=0`），避免与机械臂产生非预期干涉。

### 运行

```bash
python 02_humanoid_conveyor_sorting/humanoid_conveyor.py            # 出关键帧图片
python 02_humanoid_conveyor_sorting/humanoid_conveyor.py --viewer   # 实时 3D 窗口
python 02_humanoid_conveyor_sorting/humanoid_conveyor.py --check    # 只校验逻辑，快速
python 02_humanoid_conveyor_sorting/probe5.py                       # 复核工位 / 出料槽的 IK 可达性
```

---

## 03 灵巧手 · 掌内操纵

Shadow Hand 五指包络抓取料球 → **掌内滚动搓球** → 翻腕卸料入盘。

- **模型**：Shadow Hand E3M5（左手，**24 自由度 / 20 执行器**，MuJoCo Menagerie 官方模型）
- **场景**：自建（蓝渐变天空盒 + 棋盘格地面 + 腕部立柱 + 四壁接料盘）

### 关键帧

| | |
|---|---|
| ![张开手掌](03_dexterous_hand_manipulation/frames/01_张开手掌_24自由度.png) **① 张开手掌** — 24 自由度五指张开姿态 | ![料球落入掌窝](03_dexterous_hand_manipulation/frames/02_料球落入掌窝.png) **② 料球落入掌窝** — 手指先半屈成掌窝接球 |
| ![五指包络握紧](03_dexterous_hand_manipulation/frames/03_五指包络握紧.png) **③ 五指包络握紧** — 四指 + 拇指多点包络接触 | ![掌内滚动_正向](03_dexterous_hand_manipulation/frames/04_掌内滚动_正向.png) **④ 掌内滚动（正向）** — 四指行波把球搓动 |
| ![转过一个角度](03_dexterous_hand_manipulation/frames/05_掌内滚动_球已转过一个角度.png) **⑤ 转过一个角度** — 球面三色标记的位置变化直接可见 | ![掌内滚动_反向](03_dexterous_hand_manipulation/frames/06_掌内滚动_反向.png) **⑥ 掌内滚动（反向）** — 同一套协同反相即反向滚动 |
| ![稳定持球](03_dexterous_hand_manipulation/frames/07_稳定持球.png) **⑦ 稳定持球** | ![翻腕卸料](03_dexterous_hand_manipulation/frames/08_翻腕卸料.png) **⑧ 翻腕卸料** — 整手前倾 20° 把球倒出 |
| ![料球落入接料盘](03_dexterous_hand_manipulation/frames/09_料球落入接料盘.png) **⑨ 料球落入接料盘** | |

### 实测指标

| 指标 | 数值 |
|---|---|
| 握持接触点 | **5 ~ 6 点**（四指 + 拇指包络，滚动中随步态交替） |
| 掌内滚动累计转角 | **约 900°**（正/反向各约 450°） |
| 球在掌内的水平漂移 | **约 3 mm**（抓稳 → 滚动结束） |
| 卸料结果 | 球滚出落入接料盘，**成功** |

> "累计转角"指球体姿态在滚动过程中的**转角积分**，反映持续滚动量；"水平漂移"指球心在掌内的平移，
> 3 mm 说明球是"原地自转"而不是被推着跑。

### 技术要点

**1. 执行器映射：20 个执行器里有 4 个是"固定肌腱"执行器**

Shadow Hand 的食指有 4 个关节（J4 侧摆 / J3 近节 / J2、J1 中末节），但只有 3 个执行器：
`lh_A_FFJ0` 通过**固定肌腱**同时驱动 J2 与 J1。

```xml
<tendon>
  <fixed name="lh_FFJ0">          <!-- 其余三指同理 -->
    <joint joint="lh_FFJ2" coef="1"/>
    <joint joint="lh_FFJ1" coef="1"/>
  </fixed>
</tendon>
```

所以 **24 维的关节角不能直接当 20 维 ctrl 用**，必须按传动关系映射
（`hand_common.build_ctrl_map` 直接读模型传动信息生成，不写死索引）。

**2. 自写关节力矩控制 —— 这是本工程最关键的一步**

Menagerie 的 Shadow Hand 用 `<position kp="...">` **直接力**执行器，而 kp 只有 **0.4 ~ 1.5**，
握持力量级仅 **~1 N**，手指一搓就打滑。实测：把所有执行器力限提高 40 倍，球的转角**一个数字都不变** ——
因为瓶颈压根不是力限，而是增益。

解法：把内置执行器增益置零，改成 **24 关节 PD 力矩 + 重力/科氏补偿**（写 `qfrc_applied`）：

```python
m.actuator_gainprm[:, 0] = 0.0            # 关掉自带低增益执行器
tau = Kp * (q_des - q) - Kd * qvel + qfrc_bias   # 计算力矩
d.qfrc_applied[dof] = tau
```

**同一套协同，球的掌内转角从 8° 提升到 100°+**（12 s 内）。

**3. 掌内滚动协同：四指"行波"**

```
手指 k（FF/MF/RF/LF）的屈伸角 = 基准 + A · sin(2πft + wave·k)
```

关键是**相位随指序递进**（`wave·k`）—— 接触点沿球面**依次扫过**，才形成连续的滚动；
四指同相只会让球在掌心"原地抖"（实测同相仅 11°）。

- 拇指 `THJ1` 反向小幅摆动，提供摩擦约束、防止球被推走；
- **正/负频率即正/反向滚动**，参数由 `scan_roll_params.py` 网格扫描得到
  （净转 ~98.5°、累计 ~441°、漂移 4.4 mm / 12 s）。

**4. 为什么最后要"翻腕"才能卸料**

手掌朝上时，**张开手指球只会停在掌心**（手本身就是个托盘，实测停 11 s 纹丝不动）。
所以卸料不能靠"松手"，而是让整只手绕世界 y 轴**前倾 20°**，球顺势滚出落入接料盘 ——
这正好对应真实机械臂"端起-倾倒"的动作。

**5. 球面的三色标记只作视觉用**

球体是各向同性的，**自转肉眼完全看不出来**。所以在球面上放 4 个小色块
（`contype=conaffinity=0`，不参与碰撞、质量可忽略），滚动时色块位置变化即可直接读出姿态。

### 运行

```bash
python 03_dexterous_hand_manipulation/shadow_hand_demo.py            # 出关键帧图片
python 03_dexterous_hand_manipulation/shadow_hand_demo.py --viewer   # 实时 3D 窗口
python 03_dexterous_hand_manipulation/shadow_hand_demo.py --check    # 只校验逻辑，快速
python 03_dexterous_hand_manipulation/probe_hand.py                  # 核对执行器映射 / 关节范围 / 掌面几何
python 03_dexterous_hand_manipulation/scan_roll_params.py            # 重跑掌内滚动协同的参数网格扫描
```

---

## 04 人形 · 工位分拣

H1 站定在胸高工作台前，用左臂完成「料道送料 → 识别颜色 → 抓取 → 分入对应料箱」。

- **模型**：Unitree H1（19 DoF，MuJoCo Menagerie 官方模型）
- **场景**：自建胸高工作台 + 料道 + 红/蓝/绿三个分类料箱

### 关键帧

| | |
|---|---|
| ![传送带送料](04_humanoid_workcell_sorting/frames/01_传送带送料.png) **① 传送带送料** — 料箱沿料道送到抓取点 | ![抓取 red](04_humanoid_workcell_sorting/frames/02_抓取red.png) **② 抓取 red** — 左臂 4-DoF IK 定位 |
| ![放入 red 箱](04_humanoid_workcell_sorting/frames/03_放入red箱.png) **③ 放入 red 箱** | ![抓取 green](04_humanoid_workcell_sorting/frames/04_抓取green.png) **④ 抓取 green** |
| ![放入 green 箱](04_humanoid_workcell_sorting/frames/05_放入green箱.png) **⑤ 放入 green 箱** | ![抓取 blue](04_humanoid_workcell_sorting/frames/06_抓取blue.png) **⑥ 抓取 blue** |

### 实测指标

| 指标 | 数值 |
|---|---|
| 分拣正确性 | **4 / 4 全部正确** |
| 末端跟踪残差 | ≤ 约 **1.8 cm** |
| 入箱偏差 | ≤ **3.5 cm**（两个同色件同箱分槽，属设计值） |
| 站立稳定性 | 骨盆高度偏差 < 1 mm |

### 技术要点

**1. 工作区按「实测站姿」标定，而不是按模型默认位姿**

这里踩过一个很典型的坑：H1 的 `home` 关键帧把脚压进地面约 8 cm，
**实际稳定站立时骨盆高度是 1.06 m，而不是 keyframe 里的 0.98 m**；
同时 weld 锚点如果不显式指定，会把骨盆锁到模型编译默认位姿 `<body pos="0 0 1.06">`，
**把腿顶成外八字**（腿部关节偏差可达 0.44 rad）。

做法：把骨盆默认 z 与 `HOME_Z` 对齐（腿部偏差降到 0.007 rad），
再用探针脚本从**真实站立姿态**出发扫描可达工作区，按实测数据定工作台高度 ——
最终所有抓取点 / 料箱位的 IK 残差 < 2 cm。

**2. 两个真实的索引陷阱**

- **`jnt_qposadr` ≠ `jnt_dofadr`**：自由关节有 **7 个 qpos 但只有 6 个速度自由度**。
  写位置要用 `jnt_qposadr`，写速度必须用 `jnt_dofadr`；混用会漏清速度并越界污染相邻物体
  （料箱"凭空漂移"的根因）。同样，雅可比的列索引要用 `jnt_dofadr`。
- **`mj_jacSite` 前必须先 `mj_comPos`**，否则雅可比恒为 0，IK 原地不动。

**3. DLS 的局部极小与多初值**

同一个目标点，从"当前位姿"出发残差 5 cm，从零位出发却只有 0.1 cm。
解法是 **best-of-seeds**：当前位姿与零位各解一次，取误差更小的解。

**4. 相位推进要用「实测」手端位置判定**

原实现只看"指令到位"就切下一个相位，此时手臂其实还差几厘米没跟上，放料必然偏。
改成用 `site_xpos` 实测手端误差作为到位判据后，入箱精度稳定。

### 运行

```bash
python 04_humanoid_workcell_sorting/humanoid_sorting.py            # 出关键帧图片
python 04_humanoid_workcell_sorting/humanoid_sorting.py --viewer   # 实时 3D 窗口
python 04_humanoid_workcell_sorting/humanoid_sorting.py --check    # 只校验逻辑，快速
```

---

## 05 人形 · 运动控制

H1 的 19 自由度全身 routine：双足行走、抗扰动平衡、全身协同与定点转向 ——
不是 hello-world 式的站立演示，而是一段可直接拆解控制层次的动态 routine。

- **模型**：Unitree H1（19 自由度驱动：双腿 5×2 + 躯干 1 + 双臂 4×2）

### 关键帧

| | |
|---|---|
| ![站立](05_humanoid_motion_control/frames/01_站立.png) **① 双足站立保持** — 基座稳定器维持直立与高度 | ![行走](05_humanoid_motion_control/frames/02_行走.png) **② 双足行走步态** — 双腿反相摆动 + 重心起伏 + 摆臂协同 |
| ![抗扰动](05_humanoid_motion_control/frames/03_抗扰动.png) **③ 抗扰动平衡** — 躯干受 95N 侧向冲量后被拉回 | ![挥手](05_humanoid_motion_control/frames/04_挥手.png) **④ 全身协同挥手** — 双臂上举摆动，与下肢解耦 |
| ![转向](05_humanoid_motion_control/frames/05_转向.png) **⑤ 原地转向** — 基座偏航力矩 + 躯干联动 pivot | ![收尾](05_humanoid_motion_control/frames/06_收尾.png) **⑥ 回正收尾** — 无超调地回到竖直 |

### 实测指标

| 指标 | 数值 |
|---|---|
| 行走时基座相对竖直偏差 | 约 **2°** |
| 95 N 侧向推搡下偏差峰值 | 约 **2°**，完整恢复 |
| 骨盆高度区间 | **[0.922, 0.977]** m（home = 0.98） |
| 全程是否跌倒 | **否** |

### 技术要点

| 技术点 | 实现 |
|---|---|
| 物理仿真 | MuJoCo + Menagerie 官方 **Unitree H1** 模型（19 自由度驱动） |
| 关节控制 | 每个驱动关节 **PD 力矩控制**（腿增益强、臂增益轻），跟踪解析轨迹 |
| 平衡 / 简化 WBC | 对浮动基座（骨盆）施加**六维力/力矩**（位置 + 姿态四元数误差），规避 rpy 奇异 |
| 轨迹生成 | 行走 / 挥手 / 转向均由**解析轨迹**生成，每步物理含义可解释（非黑箱 RL） |
| 渲染 | `mujoco.Renderer` 离线渲染 + 关键帧 PNG（headless，无需显示器） |

**一个关键工程细节**：**用四元数姿态误差而不是 rpy 角误差**来做基座姿态反馈。
欧拉角在 ±90° 附近有奇异，人形在受扰恢复时会短暂经过大角度姿态，
用 rpy 会突然拿到错误的误差方向，导致控制器"抽风"；四元数误差在整个姿态空间上都连续。

### 运行

```bash
python 05_humanoid_motion_control/humanoid_demo.py            # 出关键帧图片
python 05_humanoid_motion_control/humanoid_demo.py --check    # 只跑逻辑不渲染，校验是否摔倒
```

---

## 技术对照

五个项目刻意选择了不同的机器人本体与不同的控制层次，合起来是一条完整的能力线：

| | 01 机械臂 | 02 人形·物流带 | 03 灵巧手 | 04 人形·工位 | 05 人形·运动控制 |
|---|---|---|---|---|---|
| 本体 | 7-DoF 固定基座臂 | 19-DoF 人形 | 24-DoF 多指手 | 19-DoF 人形 | 19-DoF 人形 |
| 任务层 | 抓取-放置闭环 | 队列分拣节拍 | 接触丰富掌内操纵 | 单件抓取-放置 | 行走 / 平衡 / 转向 |
| 运动学 | DLS 逆运动学（6-DoF 位姿） | 5-DoF 全身 IK（腰 + 臂） | 关节空间协同 + 接触约束 | 4-DoF 臂 IK + 多初值 | 解析轨迹（关节空间） |
| 控制层 | 位置执行器 | PD 力矩 + 重力补偿 | 自写 PD 力矩 + 重力补偿 | PD 力矩 + 六维基座稳定 | PD 力矩 + 六维基座稳定 |
| 感知接口 | `classify()` 可插拔 | `classify()` 可插拔 | —（接触即状态） | `classify()` 可插拔 | —（本体感受） |
| 核心难点 | 奇异抑制、状态隔离 | 冗余自由度分配 | 传动耦合、滚动接触 | 实测标定、索引陷阱 | 姿态奇异、抗扰恢复 |

跨项目的通用工程习惯：**IK / 规划一律在独立的 `ik_data` 上求解，绝不污染仿真 `data`**；
**场景道具与真值占位集中在构建函数里，换成真实感知或真机接口时控制层不动**。

## 快速开始

```bash
pip install mujoco numpy imageio

# 逐个跑（默认只输出关键帧图片到各自 frames/）
python 01_arm_logistics_sorting/logistics_sorting.py
python 02_humanoid_conveyor_sorting/humanoid_conveyor.py
python 03_dexterous_hand_manipulation/shadow_hand_demo.py
python 04_humanoid_workcell_sorting/humanoid_sorting.py
python 05_humanoid_motion_control/humanoid_demo.py

# 快速校验（不渲染，秒级返回结论）
python 02_humanoid_conveyor_sorting/humanoid_conveyor.py --check
python 03_dexterous_hand_manipulation/shadow_hand_demo.py --check
python 04_humanoid_workcell_sorting/humanoid_sorting.py --check
python 05_humanoid_motion_control/humanoid_demo.py --check

# 实时 3D 窗口（可拖拽视角）
python 01_arm_logistics_sorting/logistics_sorting.py --viewer
```

所有机器人的 MJCF 模型均已打包在各自子目录内，**无需下载 Menagerie 仓库**；
纯 CPU 运行，不需要 GPU 或额外硬件。

## 仓库结构

```
EmbodiedSim_Demo/
├─ 01_arm_logistics_sorting/          # 机械臂 · 物流分拣
│   ├─ logistics_sorting.py
│   ├─ frames/                        # 6 张关键帧
│   └─ panda_model/                   # Franka Emika Panda（Menagerie）
├─ 02_humanoid_conveyor_sorting/      # 人形 · 物流带分拣（腰部回转）
│   ├─ humanoid_conveyor.py
│   ├─ probe5.py
│   ├─ frames/                        # 6 张关键帧
│   └─ unitree_h1_model/              # Unitree H1（Menagerie）
├─ 03_dexterous_hand_manipulation/    # 灵巧手 · 掌内操纵
│   ├─ shadow_hand_demo.py
│   ├─ hand_common.py
│   ├─ probe_hand.py
│   ├─ scan_roll_params.py
│   ├─ frames/                        # 9 张关键帧
│   └─ shadow_hand_model/             # Shadow Hand E3M5（Menagerie）
├─ 04_humanoid_workcell_sorting/      # 人形 · 工位分拣
│   ├─ humanoid_sorting.py
│   ├─ frames/                        # 6 张关键帧
│   └─ unitree_h1_model/              # Unitree H1（Menagerie）
└─ 05_humanoid_motion_control/        # 人形 · 运动控制
    ├─ humanoid_demo.py
    ├─ frames/                        # 6 张关键帧
    └─ unitree_h1_model/              # Unitree H1（Menagerie）
```

各机器人模型授权见对应子目录内的 `LICENSE`（均来自 MuJoCo Menagerie）。
