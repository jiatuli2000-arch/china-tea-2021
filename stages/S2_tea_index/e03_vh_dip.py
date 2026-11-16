"""S2 · 预实验 E03：Sentinel-1 VH 5–8 月「先降后升」（规则见 README「预实验 E03」，运行前登记）。

输入：labels_io.load_probes()（只用 Google Earth 复核过的点）、data/interim/S1/ge_<表>_ts.json
输出：data/interim/S2/e03_features.csv、stages/S2_tea_index/E03_vh_dip.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
from labels_io import load_probes  # noqa: E402

CN = {"tea": "茶树", "evergreen_forest": "常绿林", "deciduous_forest": "落叶林", "cropland": "耕地", "mixed_forest": "混交林"}
ZN = {"jiangnan": "江南", "huanan": "华南"}
N_BOOT, SEED = 2000, 2021


def vh_feats(s1: list) -> dict:
    o = np.array([(d, v) for d, v, orb in s1 if orb == "A"], dtype=float)
    if len(o) < 10:
        return {}
    o = o[np.argsort(o[:, 0])]
    d, a = o[:, 0], o[:, 1] - np.median(o[:, 1])
    sm = pd.Series(a).rolling(3, center=True, min_periods=2).median().to_numpy()
    base = sm[(d >= 60) & (d <= 120)]
    win = (d >= 121) & (d <= 243)
    if not len(base) or not win.any():
        return {}
    i = np.flatnonzero(win)[np.argmin(sm[win])]
    after = sm[(d > d[i]) & (d <= d[i] + 60)]
    B, M = np.median(base), sm[i]
    R = after.max() if len(after) else np.nan
    return {"drop": B - M, "rise": R - M, "dip": min(B - M, R - M), "dip_doy": d[i], "n_vh": len(o)}


def s2_summer(s2: list) -> float:
    o = np.array(s2 or [[0, np.nan]], dtype=float)
    v = o[(o[:, 0] >= 152) & (o[:, 0] <= 243), 1]
    return float(np.median(v)) if len(v) else np.nan


def auc(a, b) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    r = rankdata(np.r_[a, b])[: len(a)].sum()
    return (r - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))  # 方向固定：茶树 dip 更大 → > 0.5


def boot_ci(a, b, rng) -> tuple[float, float]:
    v = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(N_BOOT)]
    return tuple(np.nanpercentile(v, [2.5, 97.5]))


def cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(X).any(axis=1)
    # 先标准化：否则 L2 正则对量纲小的夏季 NDVI 压得更狠（初版漏了，见 README E03 偏离记录）
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X[ok], y[ok],
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return roc_auc_score(y[ok], p)


def main() -> None:
    pr = load_probes()
    pr = pr[pr.ge_reviewed & pr.label.isin(CN) & pr.zone.isin(ZN)]
    rows = []
    for table, g in pr.groupby("table"):
        ts = json.loads((ROOT / f"data/interim/S1/ge_{table}_ts.json").read_text(encoding="utf-8"))["points"]
        for r in g.itertuples():
            p = ts.get(r.point_id, {})
            rows.append({"point_id": r.point_id, "zone": r.zone, "label": r.label,
                         "s_ndvi": s2_summer(p.get("s2")), **vh_feats(p.get("s1", []))})
    f = pd.DataFrame(rows)
    (ROOT / "data/interim/S2").mkdir(parents=True, exist_ok=True)
    f.to_csv(ROOT / "data/interim/S2/e03_features.csv", index=False)
    rng = np.random.default_rng(SEED)

    out = ["# E03 Sentinel-1 VH 5–8 月「先降后升」· 结果", "",
           f"> 预登记见 README「预实验 E03」。{len(f)} 个复核过的点，VH 只用升轨；"
           f"有效 dip 的点 {f.dip.notna().sum()} 个。方括号内为 bootstrap 95% 置信区间。", "",
           "## 1. 各类别 dip（dB）中位数 [四分位]", "",
           "| 茶区 | " + " | ".join(CN.values()) + " |", "|---" * (len(CN) + 1) + "|"]
    for z, zn in ZN.items():
        cells = []
        for c in CN:
            v = f[(f.zone == z) & (f.label == c)].dip.dropna()
            cells.append(f"{v.median():.2f} [{v.quantile(.25):.2f}, {v.quantile(.75):.2f}] (n={len(v)})" if len(v) else "—")
        out.append(f"| {zn} | " + " | ".join(cells) + " |")

    out += ["", "## 2. 茶树对其他类别的 dip 单特征 AUC", "", "| 茶区 | 对常绿林 | 对落叶林 | 对耕地 | 对混交林 |", "|---|---|---|---|---|"]
    res = {}
    for z, zn in ZN.items():
        t = f[(f.zone == z) & (f.label == "tea")].dip.dropna().to_numpy()
        cells = []
        for c in ["evergreen_forest", "deciduous_forest", "cropland", "mixed_forest"]:
            o = f[(f.zone == z) & (f.label == c)].dip.dropna().to_numpy()
            a = auc(t, o)
            if np.isnan(a):
                cells.append("—（样本不足）")
                continue
            lo, hi = boot_ci(t, o, rng)
            res[(z, c)] = (a, lo, hi)
            cells.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}]")
        out.append(f"| {zn} | " + " | ".join(cells) + " |")

    out += ["", "## 3. 茶树谷值日（DOY）", "", "| 茶区 | 中位数 | 四分位 | 对应日期 |", "|---|---|---|---|"]
    for z, zn in ZN.items():
        v = f[(f.zone == z) & (f.label == "tea")].dip_doy.dropna()
        dt = pd.Timestamp("2021-01-01") + pd.to_timedelta(v.median() - 1, "D")
        out.append(f"| {zn} | {v.median():.0f} | {v.quantile(.25):.0f}–{v.quantile(.75):.0f} | 约 {dt:%m-%d} |")

    out += ["", "## 4. 增量价值：茶树对常绿林，5 折交叉验证 logistic AUC", "",
            "| 茶区 | 只用夏季 NDVI | 夏季 NDVI + dip | 提高 |", "|---|---|---|---|"]
    gain = {}
    for z, zn in ZN.items():
        g = f[(f.zone == z) & f.label.isin(["tea", "evergreen_forest"])]
        y = (g.label == "tea").to_numpy().astype(int)
        a1 = cv_auc(g[["s_ndvi"]].to_numpy(), y)
        a2 = cv_auc(g[["s_ndvi", "dip"]].to_numpy(), y)
        gain[z] = a2 - a1
        out.append(f"| {zn} | {a1:.3f} | {a2:.3f} | {a2 - a1:+.3f} |")

    ok1 = all((z, "evergreen_forest") in res and res[(z, "evergreen_forest")][0] >= 0.75
              and res[(z, "evergreen_forest")][1] >= 0.65 for z in ZN)
    ok2 = all(gain[z] >= 0.03 for z in ZN)
    crop = {z: res.get((z, "cropland"), (np.nan,))[0] for z in ZN}
    out += ["", "## 5. 判定", "", "| 采用条件 | 结果 |", "|---|---|",
            f"| (1) 两茶区茶–常绿林 AUC ≥ 0.75 且下限 ≥ 0.65 | {'满足' if ok1 else '不满足'} |",
            f"| (2) 两茶区加入 dip 后交叉验证 AUC 提高 ≥ 0.03 | {'满足' if ok2 else '不满足'}（"
            + "，".join(f"{ZN[z]} {gain[z]:+.3f}" for z in ZN) + "） |",
            "| (3) 茶–耕地 AUC（单独报告） | " + "，".join(f"{ZN[z]} {crop[z]:.3f}" for z in ZN)
            + ("；< 0.6，不能单独使用" if any(v < 0.6 for v in crop.values()) else "") + " |", "",
            f"**结论：dip {'进入 TI 候选分量' if ok1 and ok2 else '暂不进入 TI 候选分量'}**。"
            "预实验，标注时可见 VH 曲线，存在循环论证风险；江北、西南齐了以后复验。"]
    (ROOT / "stages/S2_tea_index/E03_vh_dip.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
