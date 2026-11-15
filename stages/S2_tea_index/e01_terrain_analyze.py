"""S2 · E01 第二步：检验 L2A（Sen2Cor 已做地形校正）上残留的地形影响。

原预登记设计（再做 SCS+C）已撤销，原因与新判定规则见本目录 README「E01 偏离记录」。
输入：data/interim/S2/e01_{obs,points,images}.csv、data/raw/probes/jiangnan__huangshan*.csv
输出：stages/S2_tea_index/E01_terrain_correction.md、data/interim/S2/e01_features.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/interim/S2"
CLASSES = ["tea", "evergreen_forest", "deciduous_forest", "mixed_forest", "cropland"]
CN = {"tea": "茶树", "evergreen_forest": "常绿林", "deciduous_forest": "落叶林",
      "mixed_forest": "混交林", "cropland": "耕地"}
PAIRS = [("tea", "evergreen_forest"), ("tea", "deciduous_forest"), ("evergreen_forest", "deciduous_forest")]
FEATS = {"w": "冬季 NDVI", "s": "夏季 NDVI", "p10": "NDVI P10", "p90": "NDVI P90"}
RHO_MAX, AUC_GAP_MAX = 0.2, 0.05


def auc(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    v = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    return max(v, 1 - v)


def main() -> None:
    probes = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "data/raw/probes").glob("jiangnan__huangshan*.csv"))])
    lab = probes.set_index("point_id").label
    obs, pts, imgs = (pd.read_csv(D / f"e01_{n}.csv") for n in ("obs", "points", "images"))
    o = obs.merge(pts, on="point_id").merge(imgs[["img", "sza", "saa"]], on="img")
    o = o[o.cs >= 0.6].copy()
    r = np.radians
    o["cosi"] = (np.cos(r(o.slope)) * np.cos(r(o.sza))
                 + np.sin(r(o.slope)) * np.sin(r(o.sza)) * np.cos(r(o.saa - o.aspect)))
    o["ndvi"] = (o.B8 - o.B4) / (o.B8 + o.B4)
    win = o[(o.doy <= 59) | (o.doy >= 335)]
    summ = o[(o.doy >= 152) & (o.doy <= 243)]
    f = pd.DataFrame({
        "w": win.groupby("point_id").ndvi.median(), "s": summ.groupby("point_id").ndvi.median(),
        "p10": o.groupby("point_id").ndvi.quantile(0.1), "p90": o.groupby("point_id").ndvi.quantile(0.9),
        "wcos": win.groupby("point_id").cosi.mean(), "n_win": win.groupby("point_id").size(),
    }).join(lab).join(pts.set_index("point_id")[["slope", "aspect"]]).dropna(subset=["wcos", "w"])
    f.to_csv(D / "e01_features.csv")
    cut = f.wcos.median()
    f["grp"] = np.where(f.wcos < cut, "阴坡", "阳坡")

    out = ["# E01 Sentinel-2 地形影响检验 · 结果", "",
           f"> 数据：`data/interim/S2/e01_*.csv`，S2 L2A（`S2_SR_HARMONIZED`），{len(o)} 条无云观测（Cloud Score+ ≥ 0.6）。"
           f"冬季 = 1–2 月和 12 月。阴坡、阳坡按冬季平均 cos i 的中位数 {cut:.3f} 划分。"
           "原设计为什么改、判定规则怎么来的，见 README「E01 偏离记录」。", "",
           "## 0. 起因：L2A 已做地形校正 [已验证]", "",
           "同一景影像，黄山境内树木像元的 NDVI 对 hillshade 的回归斜率，L1C 为 +0.155 至 +0.405，L2A 为 −0.052 至 +0.072（4 景，见 README）。",
           "", "## 1. 残留光照依赖：冬季 NDVI 与冬季 cos i 的 Spearman ρ", "",
           "| 类别 | 点数 | ρ | 冬季 NDVI 中位数（阴坡 / 阳坡） |", "|---|---|---|---|"]
    rho = {}
    for c in CLASSES:
        g = f[f.label == c]
        if len(g) < 10:
            continue
        rho[c] = spearmanr(g.w, g.wcos)[0]
        sh, su = g[g.grp == "阴坡"].w.median(), g[g.grp == "阳坡"].w.median()
        out.append(f"| {CN[c]} | {len(g)}（阴 {(g.grp == '阴坡').sum()}） | {rho[c]:+.3f} | {sh:.3f} / {su:.3f} |")

    out += ["", "## 2. 可分性：单特征 AUC（全部 / 阴坡 / 阳坡）", "",
            "| 类别对 | " + " | ".join(FEATS.values()) + " |", "|---" * (len(FEATS) + 1) + "|"]
    gaps = []
    for x, y in PAIRS:
        cells = []
        for k in FEATS:
            v = {grp: auc(g[g.label == x][k], g[g.label == y][k])
                 for grp, g in (("全部", f), ("阴坡", f[f.grp == "阴坡"]), ("阳坡", f[f.grp == "阳坡"]))}
            gaps.append(abs(v["阴坡"] - v["阳坡"]))
            cells.append(f"{v['全部']:.3f} / {v['阴坡']:.3f} / {v['阳坡']:.3f}")
        out.append(f"| {CN[x]}–{CN[y]} | " + " | ".join(cells) + " |")

    ok1 = all(abs(rho[c]) < RHO_MAX for c in ("tea", "evergreen_forest") if c in rho)
    gap = np.nanmax(gaps)
    ok2 = gap < AUC_GAP_MAX
    out += ["", "## 3. 判定", "", "| 条件 | 结果 |", "|---|---|",
            f"| 茶树、常绿林的 \\|ρ\\| < {RHO_MAX} | {'满足' if ok1 else '不满足'} |",
            f"| 阴坡与阳坡的 AUC 差异 < {AUC_GAP_MAX} | {'满足' if ok2 else '不满足'}（最大差异 {gap:.3f}） |", "",
            f"**结论：{'L2A 的残留地形影响可以忽略，不做额外地形校正' if ok1 and ok2 else '存在不可忽略的残留地形影响，需要进一步处理'}**。"
            "只基于江南·黄山的数据，四个茶区齐了以后复验。"]
    (ROOT / "stages/S2_tea_index/E01_terrain_correction.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
