"""S2 · 预实验 E03c：单次观测的 VH 骤降，先去掉全区同期共同波动（规则见 README「预实验 E03c」）。

输入：labels_io.load_probes()、data/interim/S1/ge_<表>_ts.json
输出：data/interim/S2/e03c_features.csv、stages/S2_tea_index/E03c_vh_drop_raw.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
sys.path.insert(0, str(Path(__file__).parent))
from labels_io import load_probes  # noqa: E402
from e03_vh_dip import CN, ZN, auc, boot_ci, cv_auc, s2_summer, SEED  # noqa: E402

OTHERS = ["evergreen_forest", "deciduous_forest", "cropland", "mixed_forest"]


def dip(d: np.ndarray, e: np.ndarray, base_win, dip_win, pair: bool) -> tuple[float, float]:
    base = e[(d >= base_win[0]) & (d <= base_win[1])]
    w = np.flatnonzero((d >= dip_win[0]) & (d <= dip_win[1]))
    if not len(base) or len(w) < 2:
        return np.nan, np.nan
    if pair:
        v = (e[w[:-1]] + e[w[1:]]) / 2
        k = int(np.argmin(v)); M, i = v[k], w[k + 1]
    else:
        k = int(np.argmin(e[w])); M, i = e[w][k], w[k]
    after = np.sort(e[(d > d[i]) & (d <= d[i] + 60)])
    if len(after) < 2:
        return np.nan, d[i]
    B, R = np.median(base), after[-2]
    return min(B - M, R - M), d[i]


def main() -> None:
    pr = load_probes()
    pr = pr[pr.ge_reviewed & pr.label.isin(CN) & pr.zone.isin(ZN)]
    obs, s2 = [], {}
    for table, g in pr.groupby("table"):
        # 共同波动用整张表的全部点（含混合像元等所有复核过的点），不只主要类别
        ts = json.loads((ROOT / f"data/interim/S1/ge_{table}_ts.json").read_text(encoding="utf-8"))["points"]
        for pid, p in ts.items():
            obs += [(table, pid, d, v) for d, v, orb in p["s1"] if orb == "A"]
            s2[pid] = s2_summer(p.get("s2"))
    o = pd.DataFrame(obs, columns=["table", "point_id", "doy", "vh"])
    o["common"] = o.groupby(["table", "doy"]).vh.transform("median")
    o["e"] = o.vh - o.common
    o["e"] -= o.groupby("point_id").e.transform("median")
    n_dates = o.groupby("table").doy.nunique().to_dict()

    rows = []
    for r in pr.itertuples():
        g = o[o.point_id == r.point_id].sort_values("doy")
        d, e = g.doy.to_numpy(float), g.e.to_numpy()
        d1, D = dip(d, e, (60, 120), (121, 243), False)
        d2, _ = dip(d, e, (60, 120), (121, 243), True)
        c1, _ = dip(d, e, (213, 273), (274, 365), False)  # 控制：基线 8–9 月，谷值 10–12 月
        rows.append({"point_id": r.point_id, "zone": r.zone, "label": r.label, "s_ndvi": s2[r.point_id],
                     "dip1": d1, "dip2": d2, "dip1_ctrl": c1, "dip_doy": D, "n_vh": len(g)})
    f = pd.DataFrame(rows)
    f.to_csv(ROOT / "data/interim/S2/e03c_features.csv", index=False)
    rng = np.random.default_rng(SEED)

    out = ["# E03c 单次观测的 VH 骤降（已去掉全区同期共同波动）· 结果", "",
           f"> 预登记见 README「预实验 E03c」。{len(f)} 个复核点；升轨观测日期数 {n_dates}。不做平滑。"
           "方括号内为 bootstrap 95% 置信区间。", "",
           "## 1. 各类别特征中位数（dB）", "", "| 茶区 | 类别 | n | dip1（5–8 月） | dip2（连续两次） | dip1_ctrl（10–12 月） |",
           "|---|---|---|---|---|---|"]
    for z, zn in ZN.items():
        for c in CN:
            g = f[(f.zone == z) & (f.label == c)]
            if len(g) >= 5:
                out.append(f"| {zn} | {CN[c]} | {len(g)} | {g.dip1.median():.2f} | {g.dip2.median():.2f} | {g.dip1_ctrl.median():.2f} |")

    out += ["", "## 2. 茶树对其他类别的单特征 AUC（主特征 dip1；括号外为 dip2 的 AUC，仅供参考）", "",
            "| 茶区 | " + " | ".join(f"对{CN[c]}" for c in OTHERS) + " |", "|---" * (len(OTHERS) + 1) + "|"]
    res = {}
    for z, zn in ZN.items():
        t = f[(f.zone == z) & (f.label == "tea")]
        cells = []
        for c in OTHERS:
            g = f[(f.zone == z) & (f.label == c)]
            a = auc(t.dip1.dropna().to_numpy(), g.dip1.dropna().to_numpy())
            if np.isnan(a):
                cells.append("—")
                continue
            lo, hi = boot_ci(t.dip1.dropna().to_numpy(), g.dip1.dropna().to_numpy(), rng)
            res[(z, c)] = (a, lo, hi)
            a2 = auc(t.dip2.dropna().to_numpy(), g.dip2.dropna().to_numpy())
            cells.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}]；dip2 {a2:.3f}")
        out.append(f"| {zn} | " + " | ".join(cells) + " |")

    out += ["", "## 3. 茶树：谷值日与「夏季骤降是否大于非修剪期噪声」", "",
            "| 茶区 | 谷值日中位数（四分位） | dip1 − dip1_ctrl 中位数 | dip1 > dip1_ctrl 的比例 |", "|---|---|---|---|"]
    for z, zn in ZN.items():
        t = f[(f.zone == z) & (f.label == "tea")].dropna(subset=["dip1", "dip1_ctrl"])
        dd = t.dip1 - t.dip1_ctrl
        dt = pd.Timestamp("2021-01-01") + pd.to_timedelta(t.dip_doy.median() - 1, "D")
        out.append(f"| {zn} | {t.dip_doy.median():.0f}（{t.dip_doy.quantile(.25):.0f}–{t.dip_doy.quantile(.75):.0f}），约 {dt:%m-%d} "
                   f"| {dd.median():+.2f} | {(dd > 0).mean():.0%}（n={len(t)}） |")

    out += ["", "## 4. 增量价值：茶树对常绿林，5 折交叉验证 logistic AUC（已标准化）", "",
            "| 茶区 | 只用夏季 NDVI | 夏季 NDVI + dip1 | 提高 |", "|---|---|---|---|"]
    gain = {}
    for z, zn in ZN.items():
        g = f[(f.zone == z) & f.label.isin(["tea", "evergreen_forest"])]
        y = (g.label == "tea").to_numpy().astype(int)
        a1, a2 = cv_auc(g[["s_ndvi"]].to_numpy(), y), cv_auc(g[["s_ndvi", "dip1"]].to_numpy(), y)
        gain[z] = a2 - a1
        out.append(f"| {zn} | {a1:.3f} | {a2:.3f} | {a2 - a1:+.3f} |")

    ok1 = all((z, "evergreen_forest") in res and res[(z, "evergreen_forest")][0] >= 0.75
              and res[(z, "evergreen_forest")][1] >= 0.65 for z in ZN)
    ok2 = all(gain[z] >= 0.03 for z in ZN)
    crop = {z: res.get((z, "cropland"), (np.nan,))[0] for z in ZN}
    out += ["", "## 5. 判定", "", "| 采用条件 | 结果 |", "|---|---|",
            f"| (1) 两茶区茶–常绿林 AUC ≥ 0.75 且下限 ≥ 0.65 | {'满足' if ok1 else '不满足'}（"
            + "，".join(f"{ZN[z]} {res[(z, 'evergreen_forest')][0]:.3f}" for z in ZN) + "） |",
            f"| (2) 两茶区交叉验证 AUC 提高 ≥ 0.03 | {'满足' if ok2 else '不满足'}（"
            + "，".join(f"{ZN[z]} {gain[z]:+.3f}" for z in ZN) + "） |",
            "| (3) 茶–耕地 AUC（单独报告） | " + "，".join(f"{ZN[z]} {crop[z]:.3f}" for z in ZN) + " |", "",
            f"**结论：dip1 {'进入 TI 候选分量' if ok1 and ok2 else '暂不进入 TI 候选分量'}**。"
            "预实验；标注时可见 VH 散点，存在循环论证风险；需在信阳等新茶区复验。"]
    (ROOT / "stages/S2_tea_index/E03c_vh_drop_raw.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
