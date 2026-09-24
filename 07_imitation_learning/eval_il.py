#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模仿学习统一评测
================

两件事：

1. **闭环成功率对比**——随机 / 脚本专家 / BC / ACT-lite / DP，全部在同一批
   固定随机种子的 episode 上跑，指标包括成功率、目标误差、掉球率、动作平滑度。

2. **多模态复现度分析**——这是本工程最想说明的一件事。
   在示范数据里找一个"观测很接近、但动作分歧很大"的邻域（也就是
   p(a|obs) 真正多峰的地方），然后看三个策略各自能采出什么动作：

     * BC   是确定性映射，采多少次只有**一个**点。它学的是条件均值，
            所以会落在各模式的中间——而中间那块地方，示范数据里根本没有。
     * ACT  靠 z~N(0,I) 能采出多个点，理论上可以覆盖多个模式。
     * DP   靠不同噪声走反向扩散，能采到具体的模式。

   同时给出三个可比的量化指标（阈值 τ 取"示范动作两两距离的中位数"）：
     coverage   = 被任何策略样本"接住"的示范动作比例
     precision  = 落在示范动作附近的策略样本比例
     mode_cov   = 对示范动作做 k-means(k=4) 后，被覆盖到的簇比例

用法
----
    python eval_il.py --episodes 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist

from plot_style import (use_cjk, CLR_PRIMARY, CLR_EXPERT, CLR_ACC, CLR_WARN,
                        CLR_GRAY, CLR_SEQ)
use_cjk()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hand_common as H
from hand_env import ShadowHandReorientEnv, CTRL_DT, EPISODE_STEPS
from expert_modes import MODES, MODE_NAMES, mode_action
from il_models import (CHUNK, OBS_DIM, ACT_DIM, ChunkPolicy,
                       sample_action_distribution, BCPolicy, ACTLite,
                       DiffusionPolicyLite)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


# ==================================================================== 闭环评测
def evaluate(policy, n_ep=40, seed=10000, randomize=True, tag=""):
    env = ShadowHandReorientEnv(randomize=randomize)
    succ = drops = 0
    errs, cumus, rets, durs, jerks = [], [], [], [], []
    for i in range(n_ep):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=seed + i)
        total, prev, js, n_js = 0.0, None, 0.0, 0
        while True:
            a = policy(env, obs)
            if prev is not None:
                js += float(np.abs(np.asarray(a) - prev).sum())
                n_js += 1
            prev = np.asarray(a, float)
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        succ += int(info["success"])
        drops += int(info["dropped"])
        errs.append(info["theta_err_deg"])
        cumus.append(info["cumu_rot_deg"])
        rets.append(total)
        durs.append(info["step"] * CTRL_DT)
        jerks.append(js / max(n_js, 1))
    env.close()
    m = dict(success_rate=succ / n_ep, drop_rate=drops / n_ep,
             mean_err_deg=float(np.mean(errs)),
             mean_cumu_deg=float(np.mean(cumus)),
             mean_return=float(np.mean(rets)),
             mean_duration_s=float(np.mean(durs)),
             action_jerk=float(np.mean(jerks)))
    print(f"  {tag:16s} 成功 {m['success_rate'] * 100:5.1f}% | "
          f"误差 {m['mean_err_deg']:5.1f}° | 掉球 {m['drop_rate'] * 100:4.1f}% | "
          f"累计转角 {m['mean_cumu_deg']:6.1f}° | 动作抖动 {m['action_jerk']:.2f}",
          flush=True)
    return m


def load_policy(algo, device="cpu"):
    ck = torch.load(os.path.join(RES, f"il_{algo}.pt"), map_location=device,
                    weights_only=False)
    if algo == "bc":
        net = BCPolicy()
    elif algo == "act":
        net = ACTLite()
    else:
        net = DiffusionPolicyLite()
    net.load_state_dict(ck["state_dict"])
    return net, ck["act_mean"], ck["act_std"]


# ==================================================================== 多模态分析
def pick_neighbourhood(obs, act, ep_start, ep_len, progress=0.5,
                       n_nb=250, ep_mode=None, n_try=48, seed=0):
    """找一个"观测接近、动作分歧大"的邻域，返回 (query_obs, 邻域动作, 邻域下标)。

    三个刻意的选择：

    * **排除 query 所在的整个 episode**，只从其他 episode 里取邻居。
      专家动作 a(t) 是周期的（周期 = 1/freq ≈ 210 步），同一条轨迹上相隔整数个
      周期的两步动作**逐位相同**。若不排除，邻域里塞满这种周期性重复点，
      "动作近邻尺度 τ" 会被压到 0，覆盖率指标直接失效（踩过）。
    * **主动搜索一个多模态邻域**：给出 ep_mode 时，在进度 40%~60% 里随机试 n_try 个
      候选点，每个取最近 n_nb 个邻居，挑"恰好混 2~3 种主要模式、且这些模式的动作
      中心离得最远"的那个当靶场。不这么做，随便挑一个点，最近邻往往会全部落进
      同一个模式的密集簇里，对照退化成"单模式内的相位多样性"，看不出多模态（踩过）；
      而只按"混合模式数最多"来挑，又会混进 7 种模式、中心互相挤成一团，τ 被压到
      极小、指标全部退化成 0（也踩过）。
    * 邻域**只按观测距离取，不按模式分层** —— 我们要看的是"数据自然呈现的样子"，
      然后**如实报告**这个邻域里到底混了几种专家模式。
    """
    tt = np.maximum(obs[:, 62], 1e-6)
    prog = obs[:, 61] / tt
    cand = np.where((prog > progress - 0.10) & (prog < progress + 0.10))[0]
    if len(cand) < 10:
        cand = np.where((prog > 0.3) & (prog < 0.7))[0]

    ends = ep_start + ep_len
    ep_of = np.searchsorted(ends, np.arange(len(obs)), side="right")
    sd = obs.std(0) + 1e-6
    obs_n = obs / sd

    def neighbours(q_idx):
        d = cdist(obs_n, obs_n[q_idx][None, :])[:, 0]
        d = np.where(ep_of == ep_of[q_idx], np.inf, d)   # 排除同一条轨迹
        return np.argsort(d)[:n_nb]

    if ep_mode is None:
        q_idx = int(cand[len(cand) // 2])
        nb = neighbours(q_idx)
        return obs[q_idx], act[nb], nb

    rng = np.random.default_rng(seed)
    trials = (cand if len(cand) <= n_try
              else rng.choice(cand, n_try, replace=False))
    best = (-1.0, None, None)
    fallback = (-1.0, None, None)
    for q_idx in trials:
        nb = neighbours(int(q_idx))
        modes = ep_mode[np.searchsorted(ends, nb, side="right")]
        u, c = np.unique(modes, return_counts=True)
        # 兜底判据：混合度 = 1 - 最大单模式占比
        fb = 1.0 - float(c.max()) / len(nb)
        if fb > fallback[0]:
            fallback = (fb, int(q_idx), nb)
        # 首选判据：只有 2~3 种**主要模式**（各占 >=12%），且它们的动作中心离得越远越好。
        # 这样 τ（模式中心间距的一半）才有判别力；混进来 7 种模式的话中心互相挤在一起，
        # τ 被压得极小，覆盖率/精度会全部退化成 0，指标就没意义了（踩过）。
        keep = u[c >= 0.12 * len(nb)]
        if not (2 <= len(keep) <= 3):
            continue
        a_nb = act[nb]
        cents = np.stack([a_nb[modes == m].mean(0) for m in keep])
        spread = float(cdist(cents, cents).max())
        if spread > best[0]:
            best = (spread, int(q_idx), nb)
    if best[1] is None:
        best = fallback
    _, q_idx, nb = best
    return obs[q_idx], act[nb], nb


def multimodal_metrics(policy_samples, demo_actions, demo_modes, seed=0):
    """coverage / precision / nn_dist / mode_coverage。

    阈值 τ 怎么定，是最容易搞错的一步。试过两种，都错：

    * τ = 示范动作自身最近邻距离的中位数 -> 衡量的是**采样密度**（同一条轨迹上
      相邻时刻的动作几乎一样），算出来只有 0.007，而动作尺度是 1.0。
      于是任何策略都"接不住"任何示范动作，覆盖率恒为 0，指标失效。
    * τ = 去重后仍然如此，因为邻域里同模式的动作本身就是一坨连续稠密的点。

    正确的参照是**模式之间的距离**，不是点的密度。所以这里用已知的专家模式标签
    求各模式的动作中心，取"各中心到最近其他中心的距离中位数"的一半作 τ：
    一个样本落在某个模式的 τ 邻域内，就算落进了这个模式。
    """
    uniq_modes = np.unique(demo_modes)
    cents = np.stack([demo_actions[demo_modes == m].mean(0) for m in uniq_modes])
    demo_scale = float(np.sqrt(((demo_actions - demo_actions.mean(0)) ** 2).sum(1).mean()))
    if len(cents) >= 2:
        cc = cdist(cents, cents)
        np.fill_diagonal(cc, np.inf)
        tau = 0.5 * float(np.median(cc.min(1)))
    else:
        tau = 0.3 * demo_scale          # 单模式邻域：退回按动作尺度定

    pd_ = cdist(policy_samples, demo_actions)          # 样本 -> 示范动作
    nn = pd_.min(1)
    coverage = float(np.mean(pd_.min(0) <= tau))
    precision = float(np.mean(nn <= tau))
    mode_coverage = float(np.mean([
        (pd_[:, demo_modes == m].min(1) <= tau).any() for m in uniq_modes]))

    pc = cdist(policy_samples, cents)
    return dict(tau=tau, demo_scale=demo_scale,
                n_distinct_demo=int(len(np.unique(np.round(demo_actions, 6), axis=0))),
                n_modes_in_neighbourhood=int(len(uniq_modes)),
                modes_in_neighbourhood=[MODE_NAMES[m] for m in uniq_modes],
                inter_mode_centroid_dist=float(np.median(cc.min(1))) if len(cents) >= 2 else float("nan"),
                coverage=coverage, precision=precision,
                nn_dist=float(nn.mean()),
                mode_coverage=mode_coverage,
                min_dist_to_mode_centroid=float(pc.min(0).mean()))


def analyse_multimodality(demos, ep_mode, models, n_sample=128, seed=0):
    obs, act, es, el = demos
    q_obs, demo_act, nb = pick_neighbourhood(obs, act, es, el,
                                             ep_mode=ep_mode, seed=seed)
    demo_modes = ep_mode[np.searchsorted(es + el, nb, side="right")]

    # PCA(2) 的基底只用示范动作来定，保证三个策略画在同一坐标系里
    mu = demo_act.mean(0)
    U, S, Vt = np.linalg.svd(demo_act - mu, full_matrices=False)
    comp = Vt[:2]
    proj = lambda X: (X - mu) @ comp.T

    demo2 = proj(demo_act)
    out = {"demo2d": demo2, "demo_actions": demo_act, "query_obs": q_obs,
           "demo_modes": demo_modes}
    for algo, (net, am, asd) in models.items():
        s = sample_action_distribution(net, q_obs, am, asd, n=n_sample,
                                       seed=seed, device="cpu")
        out[algo] = dict(samples=s, samples2d=proj(s),
                         **multimodal_metrics(s, demo_act, demo_modes, seed=seed))
        out[algo]["dist_to_demo_mean"] = float(np.linalg.norm(s.mean(0) - mu))
    # 示范动作的条件均值离最近的**真实**动作有多远：说明"平均 = 谁都不是"
    out["demo_mean_dist_to_nn"] = float(
        np.linalg.norm(mu - demo_act[np.argmin(cdist(mu[None, :], demo_act)[0])]))
    out["demo_scale"] = float(np.sqrt(((demo_act - mu) ** 2).sum(1).mean()))
    out["explained_var"] = [float(S[0] ** 2 / (S ** 2).sum()),
                            float(S[1] ** 2 / (S ** 2).sum())]
    return out


def multimodal_survey(demos, ep_mode, models, n_query=20, n_sample=32, seed=0):
    """在**多个**观测点上给出更稳健的多模态统计。

    单个邻域的分析对"挑哪个点"很敏感（换一个点，覆盖率可能从 0% 直接跳到 20%），
    所以这里再补一组在 20 个随机观测点上取平均的统计：

      * mean_distinct_samples : 同一观测下策略能采出多少个**不同**动作
                                （确定性策略恒为 1，这是多模态退化的数学铁证）
      * mean_dist_to_demo     : 样本到最近真实示范动作的平均距离
      * demo_typical_nn_dist  : 邻域内示范动作彼此的最近邻距离 —— 即"多近才算
                                真的落在示范分布上"的尺子
      * ratio                 : 上面两者之比，>1 就说明样本落在"无人区"里
      * mean_modes_hit        : 样本能落进几个专家模式（阈值取该模式自身半径的 2 倍）
    """
    obs, act, es, el = demos
    ends = es + el
    ep_of = np.searchsorted(ends, np.arange(len(obs)), side="right")
    prog = obs[:, 61] / np.maximum(obs[:, 62], 1e-6)
    cand = np.where((prog > 0.35) & (prog < 0.75))[0]
    rng = np.random.default_rng(seed)
    qs = rng.choice(cand, min(n_query, len(cand)), replace=False)

    out = {}
    for algo, (net, am, asd) in models.items():
        n_dist, nn_mean, ref_scale, hit = [], [], [], []
        for qi in qs:
            d = cdist(obs, obs[qi][None, :])[:, 0]
            d = np.where(ep_of == ep_of[qi], np.inf, d)
            nb = np.argsort(d)[:250]
            na = act[nb]
            dd = cdist(na, na)
            np.fill_diagonal(dd, np.inf)
            ref_scale.append(dd.min(1).mean())
            s = np.asarray(sample_action_distribution(
                net, obs[qi], am, asd, n=n_sample, seed=1, device="cpu"))
            n_dist.append(len(np.unique(np.round(s, 6), axis=0)))
            nn_mean.append(cdist(s, na).min(1).mean())
            modes_nb = ep_mode[np.searchsorted(ends, nb, side="right")]
            h = 0
            for m in np.unique(modes_nb):
                sub = na[modes_nb == m]
                rad = float(np.median(cdist(sub, sub.mean(0)[None, :])[:, 0])) * 2
                if cdist(s, sub).min() <= rad:
                    h += 1
            hit.append(h)
        out[algo] = dict(
            mean_distinct_samples=float(np.mean(n_dist)),
            mean_dist_to_demo=float(np.mean(nn_mean)),
            demo_typical_nn_dist=float(np.mean(ref_scale)),
            ratio=float(np.mean(nn_mean) / max(np.mean(ref_scale), 1e-9)),
            mean_modes_hit=float(np.mean(hit)),
        )
    return out


# ==================================================================== 绘图
def plot_comparison(res, out_png):
    names = list(res.keys())
    metrics = [("success_rate", "任务成功率", 100, "%"),
               ("mean_err_deg", "结束时目标误差", 1, "度"),
               ("drop_rate", "掉球率", 100, "%"),
               ("action_jerk", "动作抖动（帧间变化）", 1, "")]
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.2))
    colors = [CLR_GRAY, CLR_EXPERT, CLR_PRIMARY, CLR_ACC, CLR_WARN][:len(names)]
    for j, (key, title, mul, unit) in enumerate(metrics):
        vals = [res[n][key] * mul for n in names]
        bars = ax[j].bar(range(len(names)), vals, color=colors, width=0.62)
        ax[j].set_xticks(range(len(names)))
        ax[j].set_xticklabels(names, rotation=18, ha="right", fontsize=9)
        ax[j].set_title(title)
        for b, v in zip(bars, vals):
            ax[j].text(b.get_x() + b.get_width() / 2,
                       b.get_height(), f"{v:.1f}{unit}",
                       ha="center", va="bottom", fontsize=8.5)
        ax[j].margins(y=0.18)
    fig.suptitle("掌内定向旋转 · 模仿学习闭环评测（同一批固定种子 episode）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"闭环对比图 -> {out_png}")


def plot_multimodality(mm, out_png):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    demo2 = mm["demo2d"]
    algos = ["bc", "act", "dp"]
    titles = {"bc": "BC（确定性，只有 1 个答案）",
              "act": "ACT-lite（z ~ N(0, I)）",
              "dp": "Diffusion Policy-lite（不同噪声）"}
    cols = {"bc": CLR_PRIMARY, "act": CLR_ACC, "dp": CLR_WARN}

    for j, algo in enumerate(algos):
        a = ax[j]
        a.scatter(demo2[:, 0], demo2[:, 1], s=16, c=CLR_GRAY, alpha=0.55,
                  label=f"示范动作（{len(demo2)} 条）", edgecolors="none")
        s2 = mm[algo]["samples2d"]
        uniq = np.unique(np.round(s2, 6), axis=0)
        a.scatter(s2[:, 0], s2[:, 1], s=30, c=cols[algo], alpha=0.75,
                  marker="o", label=f"{algo.upper()} 采样 {len(uniq)} 个不同点",
                  edgecolors="white", linewidths=0.5)
        a.set_title(titles[algo], fontsize=11)
        a.set_xlabel("PCA 主成分 1")
        if j == 0:
            a.set_ylabel("PCA 主成分 2")
        a.legend(fontsize=8, loc="best")

    # 右图：同一观测下"能采出多少个不同动作"（BC 恒为 1 —— 多模态退化的铁证）
    a = ax[3]
    n_uniq = [len(np.unique(np.round(mm[al]["samples2d"], 6), axis=0))
              for al in algos]
    bars = a.bar(np.arange(3), n_uniq, color=[cols[al] for al in algos], width=0.6)
    a.set_yscale("log")
    a.set_xticks(np.arange(3))
    a.set_xticklabels([al.upper() for al in algos], fontsize=9)
    a.set_ylabel("不同样本数（对数刻度）")
    a.set_title("同一观测下能采出多少个不同动作", fontsize=11)
    for b, v in zip(bars, n_uniq):
        a.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v}",
               ha="center", va="bottom", fontsize=10, fontweight="bold")
    a.set_ylim(0.6, max(n_uniq) * 6)
    a.margins(y=0.18)

    fig.suptitle("同一观测下的动作分布：BC 只能给一个均值点，ACT / DP 能给出多个样本",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"多模态对比图 -> {out_png}")


# ==================================================================== 主流程
def main():
    ap = argparse.ArgumentParser(description="模仿学习统一评测")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--demos", default="v1")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--mm-only", action="store_true",
                    help="跳过闭环评测（DP 采样慢），只重算多模态复现度分析")
    ap.add_argument("--survey", action="store_true",
                    help="额外做多观测点统计（结论更稳健）")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    z = np.load(os.path.join(RES, f"demos_{args.demos}.npz"))
    demos = (z["obs"], z["act"], z["ep_start"], z["ep_len"])

    models = {}
    for algo in ("bc", "act", "dp"):
        p = os.path.join(RES, f"il_{algo}.pt")
        if not os.path.exists(p):
            print(f"  [跳过] 缺少 {os.path.basename(p)}，先跑 train_il.py")
            continue
        models[algo] = load_policy(algo)

    if args.mm_only:
        prev = os.path.join(RES, f"il_eval_{args.tag}.json")
        res = json.load(open(prev, encoding="utf-8"))["eval"]
        print(f"（--mm-only）复用已有闭环评测：{os.path.basename(prev)}")
    else:
        print("=" * 66)
        print(f"模仿学习评测 · 每策略 {args.episodes} 个 episode（固定种子）")
        print("=" * 66)

        rng = np.random.default_rng(0)
        res = {}
        res["随机"] = evaluate(
            lambda e, o: rng.uniform(-1, 1, e.n_hand).astype(np.float32),
            args.episodes, tag="随机")
        res["脚本专家"] = evaluate(
            lambda e, o: mode_action(e, e._t * CTRL_DT, 0, 0.0),
            args.episodes, tag="脚本专家（A 套）")

        for algo, (net, am, asd) in models.items():
            pol = ChunkPolicy(net, am, asd, H=CHUNK, replan=4, steps=20, seed=777)
            res[algo.upper()] = evaluate(pol, args.episodes, tag=algo.upper())

        plot_comparison(res, os.path.join(RES, "il_comparison.png"))

    mm = None
    if models:
        print("\n多模态复现度分析……")
        mm = analyse_multimodality(demos, z["ep_mode"], models, seed=0)
        plot_multimodality(mm, os.path.join(RES, "il_multimodality.png"))
        nm = sorted(set(mm["demo_modes"].tolist()))
        print(f"  邻域 {len(mm['demo2d'])} 条示范（跨 episode 取近邻），"
              f"动作尺度 {mm['demo_scale']:.3f}")
        print(f"  该邻域混了 {len(nm)} 种专家模式：" +
              "、".join(MODE_NAMES[m] for m in nm))
        print(f"  去重后 {mm['bc']['n_distinct_demo']} 个不同动作，"
              f"近邻尺度 τ = {mm['bc']['tau']:.3f}")
        for algo in models:
            d = mm[algo]
            print(f"  {algo.upper():4s} 采样不同点数 {len(np.unique(np.round(d['samples2d'], 6), axis=0)):4d} | "
                  f"覆盖率 {d['coverage'] * 100:5.1f}% | 精度 {d['precision'] * 100:5.1f}% | "
                  f"模式覆盖 {d['mode_coverage'] * 100:5.1f}% | "
                  f"到最近示范动作 {d['nn_dist']:.3f}")
        print(f"  （示范动作的条件均值到最近真实动作 {mm['demo_mean_dist_to_nn']:.3f}"
              " —— 这就是 BC 会落在的无人区宽度）")

    survey = None
    if args.survey and models:
        print("\n多观测点统计（20 个观测点取平均，结论比单个邻域稳健）……")
        survey = multimodal_survey(demos, z["ep_mode"], models, seed=0)
        for a, d in survey.items():
            print(f"  {a.upper():4s} 不同样本数 {d['mean_distinct_samples']:6.1f} | "
                  f"到最近示范动作 {d['mean_dist_to_demo']:.3f} | "
                  f"示范典型间距 {d['demo_typical_nn_dist']:.3f} | "
                  f"倍数 {d['ratio']:7.2f} | 命中模式 {d['mean_modes_hit']:.2f}")

    out = dict(eval=res)
    if mm:
        out["multimodal"] = dict(
            n_neighbourhood=int(len(mm["demo2d"])),
            demo_action_scale=mm["demo_scale"],
            demo_mean_dist_to_nearest_action=mm["demo_mean_dist_to_nn"],
            pca_explained_var=mm["explained_var"],
            policies={a: dict(n_distinct_samples=len(
                np.unique(np.round(mm[a]["samples2d"], 6), axis=0)),
                **{k: v for k, v in mm[a].items()
                   if k not in ("samples", "samples2d")})
                for a in models},
        )
        if survey:
            out["multimodal"]["survey"] = survey
    with open(os.path.join(RES, f"il_eval_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n指标 -> results/il_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
