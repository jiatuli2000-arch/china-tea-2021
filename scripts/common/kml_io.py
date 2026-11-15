"""打点包 KML 读写（Google Earth 网页版导入/导出）。

地标名称约定：``<point_id>|<label>|<confidence>``，例如 ``HS21_33bf98fa|tea|3``。
- 生成打点包时 label 预填建议值（可能为 ``?``），confidence 留空，由标注者填写。
- 影像日期等写在地标描述里，形如 ``img=2021-03``。

用法：
    python scripts/common/kml_io.py parse  导出的.kml  输出.csv
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
NS = {"k": "http://www.opengis.net/kml/2.2"}

STYLE_COLORS = {  # KML 颜色 aabbggrr
    "tea": "ff00c853", "evergreen_forest": "ff1b5e20", "bamboo": "ff00bcd4",
    "deciduous_forest": "ff0091ff", "orchard": "ffff00ff", "cropland": "ff00ffff",
    "shrub_grass": "ff80ffcc", "built": "ff9e9e9e", "water": "ffff8000",
    "ancient_tea": "ff800080", "shaded_tea": "ff4040c0", "other": "ffffffff", "?": "ff0000ff",
}


def valid_labels() -> set[str]:
    cfg = yaml.safe_load((ROOT / "config" / "labels.yaml").read_text(encoding="utf-8"))
    return set(cfg["labels"])


def write_kml(df: pd.DataFrame, out: Path, title: str) -> None:
    """df 需含 point_id, lon, lat, label；可选 desc。"""
    def style_id(label: str) -> str:
        # 未知或带 ? 的预填标签（如 ?evergreen）统一用"待定"样式
        return "st_" + (label if label in STYLE_COLORS and label != "?" else "unk")

    styles = "".join(
        f'<Style id="{style_id(k)}"><IconStyle><color>{c}</color><scale>0.8</scale>'
        f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
        f"</IconStyle></Style>"
        for k, c in STYLE_COLORS.items()
    )
    marks = []
    for r in df.itertuples(index=False):
        sid = style_id(r.label)
        assert f'id="{sid}"' in styles, f"样式缺失：{sid}"
        desc = escape(str(getattr(r, "desc", "") or ""))
        marks.append(
            f"<Placemark><name>{r.point_id}|{r.label}|</name><description>{desc}</description>"
            f"<styleUrl>#{sid}</styleUrl><Point><coordinates>{r.lon:.7f},{r.lat:.7f},0</coordinates></Point></Placemark>"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(title)}</name>{styles}{''.join(marks)}</Document></kml>\n",
        encoding="utf-8",
    )


def parse_kml(path: Path) -> pd.DataFrame:
    """解析导出的 KML，返回 point_id, lon, lat, label, confidence, imagery_date, note, name_raw, problem。"""
    root = ET.parse(path).getroot()
    rows = []
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name = (pm.findtext("k:name", default="", namespaces=NS) or "").strip()
        desc = (pm.findtext("k:description", default="", namespaces=NS) or "").strip()
        coords = pm.findtext(".//k:Point/k:coordinates", default="", namespaces=NS).strip()
        lon, lat = (float(v) for v in coords.split(",")[:2]) if coords else (None, None)
        parts = [p.strip() for p in name.split("|")]
        parts += [""] * (3 - len(parts))
        m = re.search(r"img\s*=\s*(\d{4}(?:-\d{2})?)", desc)
        rows.append({
            "point_id": parts[0], "lon": lon, "lat": lat, "label": parts[1],
            "confidence": parts[2], "imagery_date": m.group(1) if m else "",
            "note": desc, "name_raw": name,
        })
    df = pd.DataFrame(rows)
    labels = valid_labels()

    def problem(r) -> str:
        p = []
        if r.label not in labels:
            p.append("label")
        if r.confidence not in {"1", "2", "3"}:
            p.append("confidence")
        if not r.imagery_date:
            p.append("imagery_date")
        if r.lon is None:
            p.append("coords")
        return ",".join(p)

    df["problem"] = df.apply(problem, axis=1)
    return df


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "parse":
        d = parse_kml(Path(sys.argv[2]))
        d.to_csv(sys.argv[3], index=False, encoding="utf-8-sig")
        print(f"{len(d)} 点；有问题 {int((d.problem != '').sum())} 点 → {sys.argv[3]}")
        print(d.label.value_counts().to_string())
    else:
        print(__doc__)
