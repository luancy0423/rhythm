#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rhythm.py — 个性化学习节奏：数据管道 + H1/H2/H3 分析

吃的是「节奏 Rhythm」Stage 1 应用导出的 JSON。
  · 单文件  -> 当成一个参与者（单人模式：管道自检 + 描述性 + 个体内 H1）
  · 文件夹  -> 每个 .json 一个参与者（多人模式：H1 混合模型 + H2 之间人回归）

用法:
    python analyze_rhythm.py path/to/file.json
    python analyze_rhythm.py path/to/folder/

输出:
    person_summary.csv   每位参与者的指标面板（自变量 + 招牌恢复指标 + 结果代理）
    day_panel.csv        逐日逐人的长表（给 H1 用）
    控制台打印 H1 / H2 / H3 的结果或「为何暂不可估」的诚实说明

依赖: pandas numpy scipy statsmodels  (matplotlib 可选)
"""
import sys, os, json, glob, warnings
from datetime import date
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
MIN_DAY = 1440
EPOCH = 15          # SRI 采样粒度（分钟），与前端一致
EPOCH_FILL = 14     # 单人模式下，少于这么多天只做描述、不强行建模

# ───────────────────────── 基础工具 ─────────────────────────
def didx(s):
    y, m, d = map(int, s.split("-"))
    return (date(y, m, d) - date(1970, 1, 1)).days

def hm(s):
    if not s or ":" not in s:
        return None
    h, m = s.split(":")
    try:
        return int(h) * 60 + int(m)
    except ValueError:
        return None

def cov(arr):
    arr = np.asarray([x for x in arr if x is not None], float)
    if len(arr) < 2 or arr.mean() == 0:
        return None
    return arr.std(ddof=1) / arr.mean()

def mssd(series):
    s = [x for x in series if x is not None]
    if len(s) < 3:
        return None
    s = np.asarray(s, float)
    return float(np.mean(np.diff(s) ** 2))

# ───────────────────────── 指标计算（与前端公式对齐） ─────────────────────────
def sleep_intervals(days):
    out = []
    for d, rec in days.items():
        sl = rec.get("sleep")
        if not sl:
            continue
        bed, wake = hm(sl.get("bed")), hm(sl.get("wake"))
        if bed is None or wake is None:
            continue
        idx = didx(d)
        wake_abs = wake + (MIN_DAY if wake <= bed else 0)
        out.append((idx * MIN_DAY + bed, idx * MIN_DAY + wake_abs, idx))
    return sorted(out)

def compute_sri(days):
    ints = sleep_intervals(days)
    if len(ints) < 2:
        return None
    def asleep(am):
        return any(s <= am < e for s, e, _ in ints)
    idxs = sorted({i for *_, i in ints})
    match = total = 0
    for a, b in zip(idxs, idxs[1:]):
        if b - a != 1:
            continue
        for t in range(0, MIN_DAY, EPOCH):
            if asleep(a * MIN_DAY + t) == asleep(b * MIN_DAY + t):
                match += 1
            total += 1
    if not total:
        return None
    return 100 * (2 * match / total - 1)

def compute_sjl(days):
    mids = {"work": [], "free": []}
    for rec in days.values():
        sl = rec.get("sleep")
        if not sl:
            continue
        bed, wake = hm(sl.get("bed")), hm(sl.get("wake"))
        if bed is None or wake is None:
            continue
        dur = (wake + (MIN_DAY if wake <= bed else 0)) - bed
        mid = ((bed + dur / 2) % MIN_DAY)
        mids["free" if sl.get("type") == "free" else "work"].append(mid)
    if not mids["work"] or not mids["free"]:
        return None
    return abs(np.mean(mids["free"]) - np.mean(mids["work"])) / 60.0

def state_series(days):
    e, m = [], []
    for d in sorted(days):
        st = days[d].get("state") or {}
        if isinstance(st.get("energy"), (int, float)):
            e.append(st["energy"])
        if isinstance(st.get("mood"), (int, float)):
            m.append(st["mood"])
    return e, m

def task_cov(sessions):
    by = {}
    for s in sessions:
        if s["status"] in ("done", "missed"):
            d = by.setdefault(s["date"], {"p": 0, "a": 0})
            d["p"] += s.get("plannedMin", 0)
            d["a"] += s.get("actualMin", 0) if s["status"] == "done" else 0
    ratios = [min(v["a"] / v["p"], 1.5) for v in by.values() if v["p"] > 0]
    return cov(ratios), len(ratios)

def recovery_metrics(sessions, N, K):
    """两套恢复定义:
       strict  = 仅显式标记『我补回来了』（那块漏掉的内容真被补上）
       lenient = strict 或『错过后 N 天内有任何完成』（行为兜底，易饱和）
    """
    misses = sorted([s for s in sessions if s["status"] == "missed"],
                    key=lambda s: didx(s["date"]))
    if not misses:
        return dict(msrr_strict=None, msrr_lenient=None, rec_days=None,
                    cascade=None, n_miss=0)
    done_idx = [didx(s["date"]) for s in sessions if s["status"] == "done"]
    miss_idx = [didx(s["date"]) for s in misses]
    strict, lenient, rec_times = 0, 0, []
    for s in misses:
        mi = didx(s["date"])
        if s.get("recovered") and s.get("recoveredOn"):
            strict += 1
            lenient += 1
            rec_times.append(max(0, didx(s["recoveredOn"]) - mi))
            continue
        hit = sorted(d for d in done_idx if mi < d <= mi + N)
        if hit:
            lenient += 1
            rec_times.append(hit[0] - mi)
    cascaded = sum(any(o > mi and o <= mi + K for o in miss_idx) for mi in miss_idx)
    n = len(misses)
    return dict(
        msrr_strict=strict / n,
        msrr_lenient=lenient / n,
        rec_days=float(np.median(rec_times)) if rec_times else None,
        cascade=cascaded / n,
        n_miss=n,
    )

# ───────────────────────── 加载 ─────────────────────────
def load(path):
    parts = {}
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        files = [path]
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as ex:
            print(f"  跳过 {f}: {ex}")
            continue
        pid = os.path.splitext(os.path.basename(f))[0]
        parts[pid] = data
    return parts

# ───────────────────────── 人级汇总 ─────────────────────────
def person_summary(parts):
    rows = []
    for pid, data in parts.items():
        days = data.get("days", {})
        sess = data.get("sessions", [])
        st = data.get("settings", {})
        N = st.get("recoveryWindow", 3)
        K = st.get("cascadeWindow", 2)
        e, m = state_series(days)
        tcov, tn = task_cov(sess)
        rec = recovery_metrics(sess, N, K)

        # 结果代理：把时间线分前/后两半，后半遵守率 = 「能否持续投入」
        done = sum(1 for s in sess if s["status"] == "done")
        missed = sum(1 for s in sess if s["status"] == "missed")
        adherence = done / (done + missed) if (done + missed) else None
        early_adh, late_adh, early_msrr, early_casc = split_half_metrics(sess, N, K)

        rows.append(dict(
            person=pid, n_days=len(days),
            sri=compute_sri(days), sjl=compute_sjl(days),
            energy_mssd=mssd(e), mood_mssd=mssd(m),
            task_cov=tcov, task_cov_n=tn,
            n_planned=sum(1 for s in sess if s["status"] == "planned"),
            n_done=done, n_missed=missed,
            adherence=adherence,
            msrr_strict=rec["msrr_strict"], msrr_lenient=rec["msrr_lenient"],
            rec_days=rec["rec_days"], cascade=rec["cascade"], n_miss=rec["n_miss"],
            early_adherence=early_adh, late_adherence=late_adh,
            early_msrr=early_msrr, early_cascade=early_casc,
            chronotype=st.get("chronotype"),
        ))
    return pd.DataFrame(rows)

def split_half_metrics(sessions, N, K):
    decided = [s for s in sessions if s["status"] in ("done", "missed")]
    if len(decided) < 4:
        return None, None, None, None
    decided = sorted(decided, key=lambda s: didx(s["date"]))
    idxs = [didx(s["date"]) for s in decided]
    mid = (min(idxs) + max(idxs)) / 2
    early = [s for s in decided if didx(s["date"]) <= mid]
    late = [s for s in decided if didx(s["date"]) > mid]
    def adh(g):
        d = sum(1 for s in g if s["status"] == "done")
        t = len(g)
        return d / t if t else None
    er = recovery_metrics(early, N, K)
    return adh(early), adh(late), er["msrr_strict"], er["cascade"]

# ───────────────────────── 逐日长表（给 H1） ─────────────────────────
def day_panel(parts):
    rows = []
    for pid, data in parts.items():
        days = data.get("days", {})
        sess = data.get("sessions", [])
        # 该人睡眠中点均值，用于算「当日偏离」
        mids = {}
        for d, rec in days.items():
            sl = rec.get("sleep")
            if not sl:
                continue
            bed, wake = hm(sl.get("bed")), hm(sl.get("wake"))
            if bed is None or wake is None:
                continue
            dur = (wake + (MIN_DAY if wake <= bed else 0)) - bed
            mids[d] = ((bed + dur / 2) % MIN_DAY, dur / 60.0)
        mean_mid = np.mean([v[0] for v in mids.values()]) if mids else None

        by_date = {}
        for s in sess:
            by_date.setdefault(s["date"], []).append(s)

        for d in sorted(set(list(days) + list(by_date))):
            st = days.get(d, {}).get("state") or {}
            sl_mid, sl_dur = mids.get(d, (None, None))
            day_sess = by_date.get(d, [])
            n_missed = sum(1 for s in day_sess if s["status"] == "missed")
            n_done = sum(1 for s in day_sess if s["status"] == "done")
            decided = n_missed + n_done
            rows.append(dict(
                person=pid, date=d, didx=didx(d),
                energy=st.get("energy"), mood=st.get("mood"),
                sleep_dur=sl_dur,
                mid_dev=(abs(sl_mid - mean_mid) / 60.0 if (sl_mid is not None and mean_mid is not None) else None),
                n_missed=n_missed, n_done=n_done,
                miss_today=1 if n_missed > 0 else 0,
                has_decided=1 if decided > 0 else 0,
            ))
    df = pd.DataFrame(rows).sort_values(["person", "didx"])
    # 加一列「昨日是否错过」用于看连锁
    df["prev_miss"] = df.groupby("person")["miss_today"].shift(1)
    return df

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2 or s.std(ddof=0) == 0:
        return s * 0
    return (s - s.mean()) / s.std(ddof=0)

# ───────────────────────── 假设检验 ─────────────────────────
def run_H1(panel, n_people):
    print("\n" + "=" * 64)
    print("H1 · 脆弱性：状态差的日子，更容易错过吗？")
    print("=" * 64)
    d = panel[panel["has_decided"] == 1].copy()
    d = d.dropna(subset=["miss_today"])
    for col in ["energy", "mood", "sleep_dur", "mid_dev"]:
        d[col + "_z"] = zscore(d[col])
    d = d.dropna(subset=["energy_z"])  # 至少要有精力
    if len(d) < 10 or d["miss_today"].nunique() < 2:
        print("  数据不足或没有「错过」与「未错过」两类，暂不建模。")
        print(f"  （当前可用天数 {len(d)}，错过率 "
              f"{d['miss_today'].mean():.2f} if len else —）")
        return
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    formula = "miss_today ~ energy_z + mood_z + sleep_dur_z + mid_dev_z"
    try:
        if n_people >= 3:
            model = smf.gee(formula, "person", data=d,
                            family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable()).fit()
            print("  模型: GEE 逻辑回归（按人聚类，稳健 SE）")
        else:
            model = smf.logit(formula, data=d).fit(disp=0)
            print("  模型: 逻辑回归（单人/少人，个体内）")
        print(model.summary().tables[1])
        # 数据驱动的解读（不预设结论）
        b = model.params.get("energy_z", np.nan)
        p = model.pvalues.get("energy_z", np.nan)
        sign = "负" if b < 0 else "正"
        if p < 0.05 and b < 0:
            verdict = "→ 精力越低的日子越容易错过，支持 H1。"
        elif p < 0.05 and b > 0:
            verdict = "→ 方向与 H1 相反，需检查数据。"
        else:
            verdict = "→ 尚不显著；样本更大后再看（方向为参考）。"
        print(f"  解读: energy_z 系数为{sign}（b={b:.3f}, p={p:.3f}）{verdict}")
    except Exception as ex:
        print(f"  模型未收敛（数据可能太稀疏）: {ex}")
        # 退而求其次：描述性相关
        corr = d[["miss_today", "energy_z", "sleep_dur_z", "mid_dev_z"]].corr()["miss_today"]
        print("  改用描述性相关（miss_today 与各预测变量）:")
        print(corr.to_string())

def run_H2(summ, n_people):
    print("\n" + "=" * 64)
    print("H2 · 增量效度：早期『恢复动力学』能预测后期持续投入吗？")
    print("=" * 64)
    d = summ.dropna(subset=["early_adherence", "late_adherence"]).copy()
    print(f"  可用参与者: {len(d)}")
    if n_people < 8 or len(d) < 8:
        print("  这是一个『人之间』的问题，需要足够多参与者（建议 ≥ 8–10 人）。")
        print("  当前样本不够，给出描述性预览与模型规格，待数据到位即可直接跑：")
        if len(d):
            cols = ["person", "early_adherence", "early_msrr", "early_cascade", "late_adherence"]
            print(d[cols].round(2).to_string(index=False))
        print("\n  规格: late_adherence ~ early_adherence + early_<恢复指标>")
        print("        恢复指标系数显著（控制早期遵守率后）→ 支持 H2。")
        return

    import statsmodels.formula.api as smf
    m0 = smf.ols("late_adherence ~ early_adherence", data=d).fit()
    print(f"  基线 R²(仅早期遵守率) = {m0.rsquared:.3f}\n")

    # 分别检验两个恢复指标的增量贡献
    for name, var in [("早期严格 MSRR", "early_msrr"), ("早期连锁率 cascade", "early_cascade")]:
        dd = d.dropna(subset=[var])
        if len(dd) < 8 or dd[var].std() < 1e-6:
            print(f"  · {name}: 方差不足或样本不够，跳过。")
            continue
        m1 = smf.ols(f"late_adherence ~ early_adherence + {var}", data=dd).fit()
        dR2 = m1.rsquared - smf.ols("late_adherence ~ early_adherence", data=dd).fit().rsquared
        b = m1.params[var]
        p = m1.pvalues[var]
        print(f"  · {name}: ΔR²={dR2:+.3f}, b={b:+.3f}, p={p:.3f}"
              + ("  ✓ 显著" if p < 0.05 else "  （未显著）"))
    print("\n  解读: 控制早期遵守率后，仍显著的那个恢复指标，就是 H2 的支持证据——")
    print("        即『漏了之后回不回得来 / 会不会连锁』独立于『漏了几次』预测后续投入。")

def run_H3():
    print("\n" + "=" * 64)
    print("H3 · 缓冲的调节作用（需要 Stage 2 / 干预数据）")
    print("=" * 64)
    print("  Stage 1 不直接记录『计划缓冲 slack』，因此 H3 暂不可估。")
    print("  待 Stage 2 加入每段计划的 slack 后，模型规格为:")
    print("    多层: outcome ~ variability * slack + (1|person)")
    print("    若 variability×slack 交互显著 → 高缓冲能缓冲高波动者的惩罚（支持 H3）。")
    print("  更强的因果版本: 随机分配『缓冲按波动度匹配』vs『标准』，比较组间。")

# ───────────────────────── 主流程 ─────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out, exist_ok=True)
    parts = load(path)
    if not parts:
        print("没有读到任何数据。")
        sys.exit(1)
    n_people = len(parts)
    mode = "多人" if n_people > 1 else "单人"
    print(f"\n读到 {n_people} 位参与者（{mode}模式）。")

    summ = person_summary(parts)
    panel = day_panel(parts)
    summ.to_csv(os.path.join(out, "person_summary.csv"), index=False)
    panel.to_csv(os.path.join(out, "day_panel.csv"), index=False)

    print("\n── 指标面板（每位参与者）" + "─" * 30)
    show = ["person", "n_days", "sri", "sjl", "energy_mssd", "task_cov",
            "adherence", "msrr_strict", "msrr_lenient", "cascade", "n_miss"]
    print(summ[show].round(2).to_string(index=False))

    # 严格 vs 宽松 MSRR 的方差诊断（需 ≥2 人才有意义）
    if n_people >= 2:
        sl, ll = summ["msrr_strict"].std(), summ["msrr_lenient"].std()
        print(f"\n  [诊断] MSRR 方差: 严格版 std={sl:.3f}, 宽松版 std={ll:.3f}")
        if pd.notna(ll) and ll < 0.08:
            print("  ⚠ 宽松版 MSRR 几乎无方差（饱和）——行为兜底定义太松，")
            print("    真实研究里应以『显式补回』为准（即应用里的「我补回来了」），")
            print("    或改用更有区分度的『连锁率 cascade』作为恢复动力学的主信号。")

    run_H1(panel, n_people)
    run_H2(summ, n_people)
    run_H3()

    print("\n已保存: person_summary.csv, day_panel.csv")
    if n_people == 1:
        print("提示: 单人数据只够验证管道与个体内趋势；H2/H3 需要多人/多周。")

if __name__ == "__main__":
    main()
