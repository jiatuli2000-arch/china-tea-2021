"""S1 · 把标注检查网页导出的 CSV 入库为探针表。

用法：
  python stages/S1_probes/ingest.py 导出.csv --zone jiangnan --area huangshan --province 安徽 --annotator <你> \
      [--relabel other=mixed]

判读依据：
- 网页里填了影像日期的点 → imagery_source = "Google Earth 历史影像"，imagery_date = 所填日期
- 没填的点 → imagery_source = "当前影像 + 2021 年 Sentinel-2 时序"，imagery_date = "ts2021"
  （探针点可以这样判读；S5 独立验证点必须逐点核对 2021 年历史影像）

有问题的点不入库，列在 *_problems.csv。输出 data/raw/probes/<zone>__<area>.csv，不覆盖已有文件。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from labels_io import parse_csv, problems  # noqa: E402

COLS = ["point_id", "lon", "lat", "label", "confidence", "imagery_source", "imagery_date",
        "zone", "province", "area", "annotator", "prefill", "source", "labeled_at", "note"]
SRC_TS = "当前影像 + 2021 年 Sentinel-2 时序"
SRC_GE = "Google Earth 历史影像"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--zone", required=True, choices=["jiangbei", "jiangnan", "huanan", "xinan"])
    ap.add_argument("--area", required=True)
    ap.add_argument("--province", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--relabel", action="append", default=[], metavar="旧=新",
                    help="入库时改标签代码，如 other=mixed（写进 note 留痕）")
    a = ap.parse_args()

    out = ROOT / f"data/raw/probes/{a.zone}__{a.area}.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖。新一轮标注请换 --area（如 {a.area}_r2）。")

    df = parse_csv(a.src)
    for rule in a.relabel:
        old, new = rule.split("=")
        hit = df.label == old
        df.loc[hit, "note"] = (df.loc[hit, "note"] + f" [入库改标签 {old}→{new}]").str.strip()
        df.loc[hit, "label"] = new
        print(f"改标签 {old}→{new}：{int(hit.sum())} 点")

    blank = df.imagery_date.str.strip() == ""
    df["imagery_source"] = SRC_GE
    df.loc[blank, "imagery_source"] = SRC_TS
    df.loc[blank, "imagery_date"] = "ts2021"

    df["problem"] = problems(df)
    bad, ok = df[df.problem != ""], df[df.problem == ""].copy()
    ok["zone"], ok["province"], ok["area"], ok["annotator"] = a.zone, a.province, a.area, a.annotator
    ok["confidence"] = ok.confidence.astype(int)
    for c in COLS:
        if c not in ok:
            ok[c] = ""

    out.parent.mkdir(parents=True, exist_ok=True)
    ok[COLS].to_csv(out, index=False, encoding="utf-8-sig")
    if len(bad):
        bad.to_csv(out.with_name(out.stem + "_problems.csv"), index=False, encoding="utf-8-sig")
    print(f"入库 {len(ok)} 点 → {out}；待修正 {len(bad)} 点")
    print(ok.imagery_source.value_counts().to_string())
    print(ok.label.value_counts().to_string())


if __name__ == "__main__":
    main()
