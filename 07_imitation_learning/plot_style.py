"""统一的 matplotlib 绘图样式（中文字体 + 配色 + 保存参数）。

matplotlib 默认字体 DejaVu Sans 不含 CJK 字形，中文标签会整片渲染成方框
（UserWarning: Glyph xxxx missing from font(s) DejaVu Sans）。这里按可用性
依次挑选系统中文字体，并把负号设成 ASCII，避免坐标轴出现空心方框。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体优先级：Windows 优先雅黑；其余为 Linux/macOS 常见回退
CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", "Arial Unicode MS",
]

# 配色（红涨绿跌的语义留给 K 线图，这里只用中性分类色）
CLR_PRIMARY = "#185FA5"
CLR_EXPERT = "#A32D2D"
CLR_ACC = "#0F6E56"
CLR_WARN = "#BA7517"
CLR_GRAY = "#8A8F98"
CLR_SEQ = ["#185FA5", "#0F6E56", "#BA7517", "#A32D2D", "#5B4BA8", "#8A8F98"]


def use_cjk():
    """挑选一个可用中文字体并设为默认；返回选中的字体名（找不到返回 None）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((c for c in CJK_CANDIDATES if c in available), None)
    if picked:
        plt.rcParams["font.sans-serif"] = [picked] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False      # 用 ASCII 负号，避免缺字形方框
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return picked


def darken_fig(fig_or_ax):
    """深色主题下的坐标轴可读性微调（本项目图片按浅底输出，故此处仅占位）。"""
    return fig_or_ax
