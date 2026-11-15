"""S1 验收：python stages/S1_probes/check.py

门槛（docs/01）：每茶区每个主要类别 >= 50 个置信度 >= 2 的点；字段完整；point_id 全局唯一；
标签合法；探针点不得与 S5 预留数据（china_tea_sample）混用——此处只检查本地探针表。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
LAB = yaml.safe_load((ROOT / "config/labels.yaml").read_text(encoding="utf-8"))
ZONES = yaml.safe_load((ROOT / "config/project.yaml").read_text(encoding="utf-8"))["tea_zones"]
MIN_N = 50


def main() -> int:
    files = sorted((ROOT / "data/raw/probes").glob("*__*.csv"))
    files = [f for f in files if not f.stem.endswith("_problems")]
    if not files:
        print("FAIL：没有探针表")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fails = []
    if not df.point_id.is_unique:
        fails.append(f"point_id 重复 {df.point_id.duplicated().sum()} 个")
    bad = ~df.label.isin(list(LAB["labels"]))
    if bad.any():
        fails.append(f"非法标签 {bad.sum()} 个")
    if df[["lon", "lat", "imagery_date"]].isna().any().any():
        fails.append("存在缺失坐标或影像日期")

    good = df[(df.confidence >= 2) & ~df.label.isin(LAB["unusable"])]
    tab = (good[good.label.isin(LAB["main_classes"])]
           .pivot_table(index="label", columns="zone", values="point_id", aggfunc="count")
           .reindex(index=LAB["main_classes"], columns=ZONES).fillna(0).astype(int))
    print("置信度>=2 的探针点数（行=类别，列=茶区）：")
    print(tab.to_string())
    short = [(c, z) for c in tab.index for z in tab.columns if tab.loc[c, z] < MIN_N]
    if short:
        fails.append(f"{len(short)} 个 类别×茶区 组合不足 {MIN_N}")

    print("\n" + ("PASS" if not fails else "FAIL：\n- " + "\n- ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
