"""S1 · Google Earth 复核包：对一张已入库的探针表生成「复核网页 + KML」。

固定流程第二步（docs/04）：打点工作台 → **Google Earth 复核** → 验收。
用法：python stages/S1_probes/make_ge_review.py huanan__anxi
输入：data/raw/probes/<表>.csv
输出（data/interim/S1/）：
  ge_<表>.csv / _ts.json / _viewer.html   复核网页：预填 = 工作台标签，按当前坐标重新提取曲线
  ge_<表>.kml                             导入 Google Earth 网页版：按标签分文件夹，每点含 10 m 方框
两边的点顺序一致（标签 → 点号），在 Google Earth 里按列表逐个看，网页里同步按 Y 认可或改标签。
"""
from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import build_viewer  # noqa: E402
import extract_timeseries  # noqa: E402

LABELS = yaml.safe_load((ROOT / "config/labels.yaml").read_text(encoding="utf-8"))["labels"]
COLOR = {"tea": "2e7d32", "evergreen_forest": "1565c0", "deciduous_forest": "ef6c00", "mixed_forest": "f9a825",
         "cropland": "c2185b", "mixed": "9e9e9e"}  # RGB；其余类别白色


def kml_color(rgb: str) -> str:  # KML 用 aabbggrr
    return "ff" + rgb[4:6] + rgb[2:4] + rgb[0:2]


def write_kml(df: pd.DataFrame, path: Path, title: str) -> None:
    styles, folders = [], []
    for code in df.label.unique():
        rgb = COLOR.get(code, "ffffff")
        styles.append(
            f'<Style id="s_{code}"><IconStyle><scale>0.8</scale><Icon><href>'
            f'https://earth.google.com/earth/document/icon?color={rgb}&amp;id=2000&amp;scale=4</href></Icon></IconStyle>'
            f'<LineStyle><color>{kml_color(rgb)}</color><width>2</width></LineStyle>'
            f'<PolyStyle><fill>0</fill></PolyStyle></Style>')
    defined = set(df.label.unique())
    for code, g in df.groupby("label", sort=False):
        assert code in defined
        pms = []
        for r in g.itertuples():
            dlat, dlon = 5 / 111320, 5 / (111320 * __import__("math").cos(r.lat * 3.14159265 / 180))
            ring = " ".join(f"{x:.7f},{y:.7f},0" for x, y in [(r.lon - dlon, r.lat - dlat), (r.lon + dlon, r.lat - dlat),
                                                             (r.lon + dlon, r.lat + dlat), (r.lon - dlon, r.lat + dlat),
                                                             (r.lon - dlon, r.lat - dlat)])
            pms.append(f'<Placemark><name>{escape(r.point_id)} {escape(LABELS[code]["name"])}</name>'
                       f'<description>{escape(r.desc)}</description><styleUrl>#s_{code}</styleUrl>'
                       f'<MultiGeometry><Point><coordinates>{r.lon:.7f},{r.lat:.7f},0</coordinates></Point>'
                       f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing>'
                       f'</outerBoundaryIs></Polygon></MultiGeometry></Placemark>')
        folders.append(f'<Folder><name>{escape(LABELS[code]["name"])}（{len(g)}）</name>{"".join(pms)}</Folder>')
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                    f'<name>{escape(title)}</name>{"".join(styles)}{"".join(folders)}</Document></kml>\n', encoding="utf-8")


def main(table: str) -> None:
    src = ROOT / f"data/raw/probes/{table}.csv"
    df = pd.read_csv(src, dtype={"moved_m": str}, keep_default_na=False)
    kit = f"ge_{table}"
    out = ROOT / f"data/interim/S1/{kit}.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖")
    order = {c: i for i, c in enumerate(LABELS)}
    df = df.sort_values(["label", "point_id"], key=lambda s: s.map(order) if s.name == "label" else s)

    def desc(r) -> str:
        d = f"工作台：{LABELS[r.label]['name']}，置信度 {r.confidence}"
        if r.get("moved_m"):
            d += f"，工作台里移动过 {float(r.moved_m):.0f} m（曲线已按现位置重新提取）"
        return d + (f"；备注：{r.note}" if r.note else "")
    kit_df = pd.DataFrame({"point_id": df.point_id, "lon": df.lon, "lat": df.lat, "label": df.label,
                           "source": "ge_review", "desc": df.apply(desc, axis=1)})
    kit_df.to_csv(out, index=False, encoding="utf-8-sig")
    write_kml(kit_df, out.with_suffix(".kml"), f"{table} Google Earth 复核")
    print(f"{len(kit_df)} 点 → {out}\nKML → {out.with_suffix('.kml')}")
    extract_timeseries.main(kit)
    build_viewer.main(kit)


if __name__ == "__main__":
    main(sys.argv[1])
