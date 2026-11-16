"""S1 · Google Earth 复核结果入库（固定流程第二步的收尾）。

用法：python stages/S1_probes/ingest_ge_review.py 复核导出.csv --table huanan__anxi --reviewer <你>
输出：data/raw/probes/review/<表>__ge.csv（不覆盖）。原探针表不改，验收时以复核结果为准。
verdict：ok = 认可工作台标签；changed = 改了标签。复核时又移动过的点，坐标以复核为准。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from labels_io import parse_csv, problems  # noqa: E402

COLS = ["point_id", "lon", "lat", "label", "confidence", "imagery_source", "imagery_date",
        "verdict", "moved_m", "note", "reviewer", "reviewed_at"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--table", required=True)
    ap.add_argument("--reviewer", required=True)
    a = ap.parse_args()
    out = ROOT / f"data/raw/probes/review/{a.table}__ge.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖")
    base = pd.read_csv(ROOT / f"data/raw/probes/{a.table}.csv")
    df = parse_csv(a.src)
    extra = set(df.point_id) - set(base.point_id)
    assert not extra, f"复核表里有探针表没有的点：{sorted(extra)[:5]}"

    blank = df.imagery_date.str.strip() == ""
    df["imagery_source"] = "Google Earth 历史影像"
    df.loc[blank, "imagery_source"] = "Google Earth 当前影像 + 2021 年 Sentinel-2 时序"
    df.loc[blank, "imagery_date"] = "ge+ts2021"
    df["problem"] = problems(df)
    bad = df[df.problem != ""]
    assert bad.empty, f"{len(bad)} 个点有问题：\n{bad[['point_id', 'problem']].head()}"
    df["reviewer"], df["reviewed_at"] = a.reviewer, df.labeled_at
    for c in COLS:
        if c not in df:
            df[c] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(out, index=False, encoding="utf-8-sig")
    miss = len(set(base.point_id) - set(df.point_id))
    print(f"复核 {len(df)} / {len(base)} 点 → {out}" + (f"；还有 {miss} 点没复核，验收不通过" if miss else ""))
    print(df.verdict.value_counts().to_string())
    ch = df[df.verdict == "changed"].merge(base[["point_id", "label"]], on="point_id", suffixes=("", "_工作台"))
    if len(ch):
        print("改标签（工作台 → 复核）：")
        print(ch.groupby(["label_工作台", "label"]).size().to_string())


if __name__ == "__main__":
    main()
