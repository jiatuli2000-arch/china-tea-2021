"""S2 · E03 补充（探索性，不参与采用判定）：各类别 VH 距平的类别中位数曲线。

输出：stages/S2_tea_index/fig/E03_vh_class_curves.png、data/interim/S2/e03_class_curves.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
from labels_io import load_probes  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
CLS = {"tea": ("茶树", "#2a78d6"), "evergreen_forest": ("常绿林", "#eb6834"),
       "deciduous_forest": ("落叶林", "#1baf7a"), "cropland": ("耕地", "#eda100")}
ZN = {"jiangnan": "江南 · 黄山", "huanan": "华南 · 安溪"}
BIN = 12  # 天，等于 S1 单轨重访周期


def main() -> None:
    pr = load_probes()
    pr = pr[pr.ge_reviewed & pr.label.isin(CLS) & pr.zone.isin(ZN)]
    rows = []
    for table, g in pr.groupby("table"):
        ts = json.loads((ROOT / f"data/interim/S1/ge_{table}_ts.json").read_text(encoding="utf-8"))["points"]
        for r in g.itertuples():
            o = np.array([(d, v) for d, v, orb in ts[r.point_id]["s1"] if orb == "A"], dtype=float)
            a = o[:, 1] - np.median(o[:, 1])
            for d, v in zip(o[:, 0], a):
                rows.append((r.zone, r.label, r.point_id, int((d - 1) // BIN), v))
    df = pd.DataFrame(rows, columns=["zone", "label", "point_id", "bin", "a"])
    per_pt = df.groupby(["zone", "label", "point_id", "bin"]).a.median().reset_index()
    cur = per_pt.groupby(["zone", "label", "bin"]).a.agg(
        med="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), n="size").reset_index()
    cur["doy"] = cur["bin"] * BIN + BIN / 2
    cur.to_csv(ROOT / "data/interim/S2/e03_class_curves.csv", index=False)

    month_doy = [pd.Timestamp(2021, m, 1).day_of_year for m in range(1, 13)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, (z, zn) in zip(axes, ZN.items()):
        ax.axvspan(121, 243, color="#000000", alpha=0.04, lw=0)
        ax.text(182, 1.45, "5–8 月", ha="center", fontsize=9, color="#6b6b66")
        for c, (name, col) in CLS.items():
            g = cur[(cur.zone == z) & (cur.label == c)]
            if g.empty:
                continue
            if c == "tea":
                ax.fill_between(g.doy, g.q25, g.q75, color=col, alpha=0.12, lw=0)
            ax.plot(g.doy, g.med, color=col, lw=2.6 if c == "tea" else 1.8)
            n = per_pt[(per_pt.zone == z) & (per_pt.label == c)].point_id.nunique()
            ax.text(g.doy.iloc[-1] + 4, g.med.iloc[-1], f"{name}（{n}）", color="#1d1d1b", fontsize=9, va="center")
        ax.axhline(0, color="#d0d0ca", lw=0.8, zorder=0)
        ax.set_xticks(month_doy, [f"{m}月" for m in range(1, 13)], fontsize=8.5)
        ax.set_xlim(1, 400)
        ax.set_ylim(-2, 1.6)
        ax.set_title(zn, loc="left", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b5b5ae")
        ax.tick_params(colors="#6b6b66")
        ax.grid(axis="y", color="#ececE8", lw=0.6)
    axes[0].set_ylabel("VH 距平（dB，减去每点全年中位数）")
    fig.suptitle("2021 年 Sentinel-1 VH（升轨）类别中位数曲线；茶树阴影 = 四分位范围；12 天分箱", fontsize=10.5, x=0.01, ha="left")
    out = ROOT / "stages/S2_tea_index/fig/E03_vh_class_curves.png"
    fig.savefig(out, dpi=160)
    print(out)
    tea = cur[cur.label == "tea"]
    for z in ZN:
        t = tea[tea.zone == z].set_index("doy").med
        print(z, "茶树中位曲线：3–4月", round(t[(t.index >= 60) & (t.index <= 120)].median(), 2),
              " 5–8月最低", round(t[(t.index >= 121) & (t.index <= 243)].min(), 2),
              "@DOY", t[(t.index >= 121) & (t.index <= 243)].idxmin(),
              " 9–10月", round(t[(t.index >= 244) & (t.index <= 304)].median(), 2))


if __name__ == "__main__":
    main()
