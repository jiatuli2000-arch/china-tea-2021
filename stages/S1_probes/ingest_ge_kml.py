"""S1 · 从 Google Earth 网页版导出的 KML 读取复核结果并入库（复核的 KML 路线）。

在 Google Earth 里复核时，任选一种方式改标签：
  - 改说明文字里的类别名：「工作台：茶树，…」→「工作台：混合像元，…」（Google Earth 会把说明挪进样式的气泡文字，脚本两处都读）
  - 改点名里的类别名：「AX21_0001 茶树」→「AX21_0001 耕地」
  - 改图标颜色为另一类的颜色（颜色表见 make_ge_review.COLOR）
  - 删除点 = 这个点不要了（verdict = removed，下游不用）
几种方式给出的类别互相矛盾时报错。没动的点 verdict = ok。

用法：python stages/S1_probes/ingest_ge_kml.py 导出.kml --table huanan__anxi --reviewer ZGF [--fix 点号=类别 ...] [--drop 点号 ...]
  --fix / --drop：复核后再次确认时的更正或删除（写进 note）。
输出：data/raw/probes/review/<表>__ge.csv（不覆盖），格式同 ingest_ge_review.py。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from make_ge_review import COLOR  # noqa: E402

NS = {"k": "http://www.opengis.net/kml/2.2"}
KID = "{http://www.opengis.net/kml/2.2}id"
LABELS = yaml.safe_load((ROOT / "config/labels.yaml").read_text(encoding="utf-8"))["labels"]
NAME2CODE = {v["name"]: c for c, v in LABELS.items()}
NAME2CODE.update({"常绿林": "evergreen_forest", "混交林": "mixed_forest", "常绿落叶混交": "mixed_forest",
                  "常绿林落叶林混交": "mixed_forest", "混合像元": "mixed", "落叶": "deciduous_forest"})
RGB2CODE = {v: k for k, v in COLOR.items()}
COLS = ["point_id", "lon", "lat", "label", "confidence", "imagery_source", "imagery_date",
        "verdict", "moved_m", "note", "reviewer", "reviewed_at"]


def to_code(name: str) -> str:
    name = name.strip()
    if name in LABELS:
        return name
    if name not in NAME2CODE:
        raise SystemExit(f"认不出的类别名「{name}」：请改成 config/labels.yaml 里的中文名")
    return NAME2CODE[name]


def parse(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    icon, balloon, smap = {}, {}, {}
    for el in root.iter():
        sid = el.get("id") or el.get(KID)
        if not sid:
            continue
        if el.tag.endswith("StyleMap"):
            for p in el.findall("k:Pair", NS):
                if p.findtext("k:key", namespaces=NS) == "normal":
                    smap[sid] = p.findtext("k:styleUrl", namespaces=NS).lstrip("#")
            continue
        h = el.find(".//k:IconStyle/k:Icon/k:href", NS)
        if h is not None and (m := re.search(r"color=([0-9a-fA-F]{6})", h.text or "")):
            icon[sid] = m.group(1).lower()
        b = el.find(".//k:BalloonStyle/k:text", NS)
        if b is not None and b.text:
            balloon[sid] = b.text
    rows = []
    for pm in root.iter(f"{{{NS['k']}}}Placemark"):
        name = pm.findtext("k:name", namespaces=NS) or ""
        sid = (pm.findtext("k:styleUrl", namespaces=NS) or "").lstrip("#")
        sid = smap.get(sid, sid)
        desc = pm.findtext("k:description", namespaces=NS) or balloon.get(sid, "")
        lon, lat = map(float, pm.find(".//k:Point/k:coordinates", NS).text.strip().split(",")[:2])
        pid, _, nm = name.partition(" ")
        m = re.match(r"工作台：([^，,]+)", desc)
        rows.append({"point_id": pid, "lon": lon, "lat": lat, "by_name": to_code(nm) if nm else None,
                     "by_desc": to_code(m.group(1)) if m else None, "by_color": RGB2CODE.get(icon.get(sid))})
    return pd.DataFrame(rows)


def ndvi_flags(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """复核后标签与 2021 年 S2 NDVI 明显矛盾的点（只提示，不改）。
    规则：常绿林/茶树冬季 NDVI < 0.45；落叶林冬季 > 0.6 或夏季 < 0.6；耕地全年振幅 < 0.2。"""
    ts = json.loads((ROOT / f"data/interim/S1/ge_{table}_ts.json").read_text(encoding="utf-8"))["points"]

    def feat(pid):
        o = np.array(ts.get(pid, {}).get("s2") or [[0, np.nan]])
        d, v = o[:, 0], o[:, 1]
        w, sm = v[(d <= 59) | (d >= 335)], v[(d >= 152) & (d <= 243)]
        return pd.Series({"冬季": np.median(w) if len(w) else np.nan, "夏季": np.median(sm) if len(sm) else np.nan,
                          "振幅": np.nanpercentile(v, 90) - np.nanpercentile(v, 10)})
    f = df[df.verdict != "removed"][["point_id", "label"]].copy()
    f = f.join(f.point_id.apply(feat))
    bad = ((f.label.isin(["evergreen_forest", "tea"]) & (f.冬季 < 0.45))
           | ((f.label == "deciduous_forest") & ((f.冬季 > 0.6) | (f.夏季 < 0.6)))
           | ((f.label == "cropland") & (f.振幅 < 0.2)))
    return f[bad].round(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kml", type=Path)
    ap.add_argument("--table", required=True)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--fix", action="append", default=[], metavar="点号=类别")
    ap.add_argument("--drop", action="append", default=[], metavar="点号",
                    help="复核后再确认要删除的点（verdict = removed，写进 note）")
    ap.add_argument("--date", default=pd.Timestamp.now().strftime("%Y-%m-%d"), help="复核日期")
    a = ap.parse_args()
    out = ROOT / f"data/raw/probes/review/{a.table}__ge.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖")
    base = pd.read_csv(ROOT / f"data/raw/probes/{a.table}.csv", dtype={"moved_m": str}, keep_default_na=False)
    k = parse(a.kml)
    assert k.point_id.is_unique, "KML 里点号重复"
    extra = set(k.point_id) - set(base.point_id)
    assert not extra, f"KML 里有探针表没有的点：{sorted(extra)[:5]}"
    df = base[["point_id", "lon", "lat", "label", "confidence"]].merge(k, on="point_id", how="left",
                                                                        suffixes=("_b", ""))
    removed = df.by_desc.isna() & df.by_name.isna() & df.lon.isna()

    def decide(r) -> str:
        if pd.isna(r.lon):
            return ""
        changed = {v for v in (r.by_name, r.by_desc, r.by_color) if isinstance(v, str) and v != r.label}
        if len(changed) > 1:
            raise SystemExit(f"{r.point_id}：名字/说明/颜色给出的类别矛盾 {changed}")
        return changed.pop() if changed else r.label
    df["new"] = df.apply(decide, axis=1)
    df["note"] = ""
    for f in a.fix:
        pid, lab = f.split("=")
        assert pid in set(df.point_id), pid
        assert lab in LABELS, lab
        hit = df.point_id == pid
        df.loc[hit, "note"] = f"复核后再确认：{df.loc[hit, 'new'].iat[0] or '已删除'}→{lab}"
        df.loc[hit, ["new", "lon", "lat"]] = [lab, df.loc[hit, "lon_b"].iat[0], df.loc[hit, "lat_b"].iat[0]]
        removed &= ~hit
    for pid in a.drop:
        assert pid in set(df.point_id), pid
        hit = df.point_id == pid
        df.loc[hit, "note"] = "复核后再确认删除（标签与 2021 年 NDVI 矛盾）"
        df.loc[hit, ["lon", "lat"]] = [np.nan, np.nan]
        removed |= hit
    df["verdict"] = "ok"
    df.loc[df.new != df.label, "verdict"] = "changed"
    df.loc[removed, "new"] = df.loc[removed, "label"]
    df.loc[removed, "verdict"] = "removed"
    mv = ((df.lon - df.lon_b).abs() + (df.lat - df.lat_b).abs()) > 1e-6
    assert not mv[~removed].any(), f"有点被拖动过，KML 路线不处理移动：{df.point_id[mv].tolist()[:5]}"
    df["lon"], df["lat"] = df.lon_b, df.lat_b
    df["label"] = df.new
    df["imagery_source"], df["imagery_date"] = "Google Earth 网页版当前影像（复核）", "ge"
    df["moved_m"], df["reviewer"], df["reviewed_at"] = "", a.reviewer, a.date
    flags = ndvi_flags(df, a.table)
    if len(flags):
        print(f"与 2021 年 NDVI 明显矛盾的点 {len(flags)} 个（请在 Google Earth 再看一眼，需要就用 --fix 更正后重跑）：")
        print(flags.to_string(index=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}：{len(df)} 点")
    print(df.verdict.value_counts().to_string())
    ch = df[df.verdict != "ok"].merge(base[["point_id", "label"]], on="point_id", suffixes=("", "_工作台"))
    if len(ch):
        print(ch[["point_id", "label_工作台", "label", "verdict", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
