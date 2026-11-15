"""S2 · E01b 第二步：bootstrap 置信区间 + 经验校正前后对比（规则见 README「预实验 E01b」）。

输入：data/interim/S2/e01_{obs,points,images}.csv、e01b_beta.csv、data/raw/probes/jiangnan__huangshan*.csv
输出：stages/S2_tea_index/E01b_residual_terrain.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/interim/S2"
CN = {"tea": "茶树", "evergreen_forest": "常绿林", "deciduous_forest": "落叶林", "cropland": "耕地"}
PAIRS = [("tea", "evergreen_forest"), ("tea", "deciduous_forest"), ("evergreen_forest", "deciduous_forest")]
FEATS = {"w": "冬季 NDVI", "s": "夏季 NDVI", "p10": "NDVI P10", "p90": "NDVI P90"}
N_BOOT, SEED, N_MIN = 2000, 2021, 300


def auc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    r = rankdata(np.r_[a, b])[: len(a)].sum()
    v = (r - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return max(v, 1 - v)


def load() -> pd.DataFrame:
    probes = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "data/raw/probes").glob("jiangnan__huangshan*.csv"))])
    lab = probes.set_index("point_id").label
    obs, pts, imgs = (pd.read_csv(D / f"e01_{n}.csv") for n in ("obs", "points", "images"))
    beta = pd.read_csv(D / "e01b_beta.csv")
    beta["month"] = pd.to_datetime(beta.img.str[:8]).dt.month
    good = beta[beta.n >= N_MIN]
    by_month = good.groupby("month").beta.median()
    beta["b_use"] = np.where(beta.n >= N_MIN, beta.beta,
                             beta.month.map(by_month).fillna(good.beta.median()))
    beta["b_src"] = np.where(beta.n >= N_MIN, "本景", np.where(beta.month.isin(by_month.index), "同月中位数", "全年中位数"))

    o = (obs.merge(pts, on="point_id").merge(imgs[["img", "sza", "saa"]], on="img")
         .merge(beta[["img", "b_use", "b_src"]], on="img"))
    o = o[o.cs >= 0.6].copy()
    r = np.radians
    o["cos_z"] = np.cos(r(o.sza))
    o["cosi"] = (np.cos(r(o.slope)) * o.cos_z + np.sin(r(o.slope)) * np.sin(r(o.sza)) * np.cos(r(o.saa - o.aspect)))
    o["ndvi"] = (o.B8 - o.B4) / (o.B8 + o.B4)
    o["ndvi_c"] = o.ndvi - o.b_use * (o.cosi - o.cos_z)
    load.meta = {"n_obs": len(o), "beta": beta, "good": good, "src": o.b_src.value_counts().to_dict()}

    win = o[(o.doy <= 59) | (o.doy >= 335)]
    summ = o[(o.doy >= 152) & (o.doy <= 243)]
    parts = {"wcos": win.groupby("point_id").cosi.mean()}
    for tag, col in (("raw", "ndvi"), ("cor", "ndvi_c")):
        parts[f"w_{tag}"] = win.groupby("point_id")[col].median()
        parts[f"s_{tag}"] = summ.groupby("point_id")[col].median()
        parts[f"p10_{tag}"] = o.groupby("point_id")[col].quantile(0.1)
        parts[f"p90_{tag}"] = o.groupby("point_id")[col].quantile(0.9)
    f = pd.DataFrame(parts).join(lab).dropna(subset=["wcos", "w_raw"])
    f = f[f.label.isin(CN)]
    f["grp"] = np.where(f.wcos < f.wcos.median(), "阴坡", "阳坡")
    return f


def stats(f: pd.DataFrame, tag: str) -> dict:
    out = {}
    for c in ("tea", "evergreen_forest"):
        g = f[f.label == c]
        out[("rho", c)] = spearmanr(g[f"w_{tag}"], g.wcos)[0]
    for x, y in PAIRS:
        for k in FEATS:
            col = f"{k}_{tag}"
            v = {}
            for grp, g in (("全部", f), ("阴坡", f[f.grp == "阴坡"]), ("阳坡", f[f.grp == "阳坡"])):
                v[grp] = auc(g.loc[g.label == x, col].to_numpy(), g.loc[g.label == y, col].to_numpy())
            out[("auc", x, y, k)] = v["全部"]
            out[("gap", x, y, k)] = v["阴坡"] - v["阳坡"]
    return out


def main() -> None:
    f = load()
    rng = np.random.default_rng(SEED)
    strata = [g.index.to_numpy() for _, g in f.groupby(["label", "grp"])]
    point = {t: stats(f, t) for t in ("raw", "cor")}
    boot = {t: {k: [] for k in point[t]} for t in ("raw", "cor")}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        fb = f.loc[idx]
        for t in ("raw", "cor"):
            for k, v in stats(fb, t).items():
                boot[t][k].append(v)
    ci = {t: {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in boot[t].items()} for t in boot}

    def fmt(t, k):
        lo, hi = ci[t][k]
        return f"{point[t][k]:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    m = load.meta
    out = ["# E01b 残留地形影响：置信区间与经验校正 · 结果", "",
           f"> {m['n_obs']} 条无云观测，bootstrap {N_BOOT} 次（按类别 × 阴阳坡分层）。方括号为 95% 置信区间。",
           f"> β：{len(m['beta'])} 景中 {len(m['good'])} 景有效像元 ≥ {N_MIN}，β 中位数 {m['good'].beta.median():+.3f}"
           f"（四分位 {m['good'].beta.quantile(.25):+.3f} 到 {m['good'].beta.quantile(.75):+.3f}）。观测所用 β 来源：{m['src']}。", "",
           "## 1. 冬季 NDVI 与冬季 cos i 的 ρ", "", "| 类别 | 校正前 | 校正后 |", "|---|---|---|"]
    for c in ("tea", "evergreen_forest"):
        out.append(f"| {CN[c]} | {fmt('raw', ('rho', c))} | {fmt('cor', ('rho', c))} |")

    def excl0(t, k):
        lo, hi = ci[t][k]
        return lo > 0 or hi < 0
    out += ["", "## 2. AUC 阴坡 − 阳坡（置信区间不含 0 的加 ★）", "",
            "| 类别对 · 特征 | 校正前 | 校正后 |", "|---|---|---|"]
    n_ex = {"raw": 0, "cor": 0}
    for x, y in PAIRS:
        for k, nm in FEATS.items():
            key = ("gap", x, y, k)
            cells = []
            for t in ("raw", "cor"):
                star = excl0(t, key)
                n_ex[t] += star
                cells.append(fmt(t, key) + (" ★" if star else ""))
            out.append(f"| {CN[x]}–{CN[y]} · {nm} | " + " | ".join(cells) + " |")
    out += ["", "## 3. 全部点的 AUC", "", "| 类别对 · 特征 | 校正前 | 校正后 | 变化 |", "|---|---|---|---|"]
    drops = []
    for x, y in PAIRS:
        for k, nm in FEATS.items():
            key = ("auc", x, y, k)
            d = point["cor"][key] - point["raw"][key]
            if x == "tea":
                drops.append(d)
            out.append(f"| {CN[x]}–{CN[y]} · {nm} | {fmt('raw', key)} | {fmt('cor', key)} | {d:+.3f} |")

    real = any(excl0("raw", ("rho", c)) for c in ("tea", "evergreen_forest")) or n_ex["raw"] > 0
    ok1 = all(abs(point["cor"][("rho", c)]) <= abs(point["raw"][("rho", c)]) - 0.05 for c in ("tea", "evergreen_forest"))
    ok2 = min(drops) >= -0.02
    ok3 = n_ex["cor"] <= n_ex["raw"]
    out += ["", "## 4. 判读", "",
            f"- 残留是否真实：ρ 置信区间不含 0 → 茶树 {'是' if excl0('raw', ('rho', 'tea')) else '否'}、常绿林 {'是' if excl0('raw', ('rho', 'evergreen_forest')) else '否'}；"
            f"AUC 阴阳坡差置信区间不含 0 的项：{n_ex['raw']} / {len(PAIRS) * len(FEATS)}。",
            "", "| 采用经验校正的条件 | 结果 |", "|---|---|",
            f"| (i) 茶树、常绿林的 \\|ρ\\| 都下降 ≥ 0.05 | {'满足' if ok1 else '不满足'} |",
            f"| (ii) 茶–其他各项 AUC 下降 ≤ 0.02 | {'满足' if ok2 else '不满足'}（最大变化 {min(drops):+.3f}） |",
            f"| (iii) 阴阳坡差显著项数没有增加 | {'满足' if ok3 else '不满足'}（{n_ex['raw']} → {n_ex['cor']}） |", "",
            f"**结论：{'采用经验地形校正' if ok1 and ok2 and ok3 else '不采用经验地形校正'}**；"
            f"残留效应{'在统计上真实存在' if real else '与抽样噪声无法区分'}。只基于江南·黄山。"]
    (ROOT / "stages/S2_tea_index/E01b_residual_terrain.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
