"""S1 · 移动点复核：把标注时挪到纯净像元的点按新位置重新提取时序，生成复核网页。

用法：python stages/S1_probes/review_moved.py anxi_kit_v1 data/raw/probes/exports/<导出>.csv
输出：data/interim/S1/<kit>_moved.csv、_ts.json、_viewer.html
  预填 = 你第一次标的标签，desc 写明移动距离。复核网页导出的 CSV 在 ingest.py 里用 --review 传入。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import build_viewer  # noqa: E402
import extract_timeseries  # noqa: E402


def main(kit: str, export: str) -> None:
    ex = pd.read_csv(export, dtype=str, keep_default_na=False)
    if "moved_m" not in ex or not (ex.moved_m != "").any():
        sys.exit("导出里没有移动过的点，不需要复核")
    mv = ex[ex.moved_m != ""]
    desc = pd.read_csv(ROOT / f"data/interim/S1/{kit}.csv").set_index("point_id").desc
    new = f"{kit}_moved"
    out = ROOT / f"data/interim/S1/{new}.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖")
    pd.DataFrame({"point_id": mv.point_id, "lon": mv.lon.astype(float), "lat": mv.lat.astype(float),
                  "label": mv.label, "source": "moved",
                  "desc": [f"已移动 {float(m):.0f} m，曲线为新位置。原位置：{desc.get(p, '')}"
                           for p, m in zip(mv.point_id, mv.moved_m)]}
                 ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(mv)} 个移动点 → {out}")
    extract_timeseries.main(new)
    build_viewer.main(new)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
