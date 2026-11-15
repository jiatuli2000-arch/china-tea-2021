"""S2 · E01 第一步：从 GEE 取地形校正实验所需的原始数据（不做任何判断）。

输出（data/interim/S2/）：
  e01_obs.csv     每点每景：point_id, img, doy, B4, B8, cs
  e01_points.csv  每点：point_id, slope, aspect（度）
  e01_images.csv  每景：img, doy, sza, saa（太阳天顶角/方位角，度），C_B4, C_B8, n_fit
"""
from __future__ import annotations

import os
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
YEAR = yaml.safe_load((ROOT / "config/project.yaml").read_text(encoding="utf-8"))["year"]
BOUNDARY = "projects/project-c7b654cb-9d08-48ed-bcf/assets/01_boundary/study_areas/Huangshan_Boundary"
OUT = ROOT / "data/interim/S2"
BATCH = 15  # 每批点数：点数 × 景数不能超过 GEE 单次 5000 条上限


def terrain():
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection())
    return ee.Terrain.products(dem).select(["slope", "aspect"])


def cos_i(img: ee.Image, ter: ee.Image) -> ee.Image:
    rad = ee.Number(3.141592653589793).divide(180)
    sz = ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE")).multiply(rad)
    sa = ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")).multiply(rad)
    s = ter.select("slope").multiply(rad)
    a = ter.select("aspect").multiply(rad)
    return (s.cos().multiply(sz.cos())
            .add(s.sin().multiply(sz.sin()).multiply(a.multiply(-1).add(sa).cos()))).rename("cosi")


def main() -> None:
    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    OUT.mkdir(parents=True, exist_ok=True)
    probes = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "data/raw/probes").glob("jiangnan__huangshan*.csv"))])
    probes = probes[(probes.confidence >= 2) &
                    probes.label.isin(["tea", "evergreen_forest", "deciduous_forest", "mixed_forest", "cropland"])]
    print("实验点数：", probes.label.value_counts().to_dict())

    region = ee.FeatureCollection(BOUNDARY).geometry()
    ter = terrain()
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").linkCollection(cs, ["cs_cdf"]))
    tree = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10)

    # 1) 每景 C 系数：该景黄山境内、无云、树木像元上，反射率 ~ cos i 的线性回归
    def per_image(img):
        ci = cos_i(img, ter)
        stack = (img.select(["B4", "B8"]).divide(10000).addBands(ci)
                 .updateMask(tree).updateMask(img.select("cs_cdf").gte(0.6)))
        smp = stack.sample(region=region, scale=60, numPixels=3000, seed=1, tileScale=4)
        fit4 = smp.reduceColumns(ee.Reducer.linearFit(), ["cosi", "B4"])
        fit8 = smp.reduceColumns(ee.Reducer.linearFit(), ["cosi", "B8"])
        return ee.Feature(None, {
            "img": img.get("system:index"), "doy": img.date().getRelative("day", "year").add(1),
            "sza": img.get("MEAN_SOLAR_ZENITH_ANGLE"), "saa": img.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
            "m_B4": fit4.get("scale"), "b_B4": fit4.get("offset"),
            "m_B8": fit8.get("scale"), "b_B8": fit8.get("offset"), "n_fit": smp.size()})
    ids = col.aggregate_array("system:index").getInfo()
    print(f"{YEAR} 年覆盖黄山的 S2 景数：{len(ids)}")
    rows = []
    for i in (range(0, len(ids), 20) if not (OUT / "e01_images.csv").exists() else []):
        sub = col.filter(ee.Filter.inList("system:index", ids[i:i + 20]))
        rows += [f["properties"] for f in ee.FeatureCollection(sub.map(per_image)).getInfo()["features"]]
        print(f"C 系数 {min(i + 20, len(ids))}/{len(ids)}", flush=True)
    if rows:
        imgs = pd.DataFrame(rows)
        for b in ("B4", "B8"):
            imgs[f"C_{b}"] = imgs[f"b_{b}"] / imgs[f"m_{b}"]
        imgs.to_csv(OUT / "e01_images.csv", index=False)
    imgs = pd.read_csv(OUT / "e01_images.csv")

    # 2) 每点地形
    pts_all = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
                                    for r in probes.itertuples()])
    t = ter.sampleRegions(pts_all, scale=30).getInfo()["features"]
    pd.DataFrame([f["properties"] for f in t]).to_csv(OUT / "e01_points.csv", index=False)

    # 3) 每点每景原始反射率
    obs = []
    for i in range(0, len(probes), BATCH):
        part = probes.iloc[i:i + BATCH]
        pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
                                    for r in part.itertuples()])
        def per(img):
            fc = img.select(["B4", "B8", "cs_cdf"]).rename(["B4", "B8", "cs"]).reduceRegions(pts, ee.Reducer.first(), 10)
            return fc.filter(ee.Filter.notNull(["B4"])).map(lambda f: f.set("img", img.get("system:index")))
        flat = col.filterBounds(pts.geometry()).map(per).flatten().select(["point_id", "img", "B4", "B8", "cs"], None, False)
        obs += [f["properties"] for f in flat.getInfo()["features"]]
        print(f"观测 {min(i + BATCH, len(probes))}/{len(probes)}", flush=True)
    o = pd.DataFrame(obs)
    o[["B4", "B8"]] = o[["B4", "B8"]] / 10000
    o.merge(imgs[["img", "doy"]], on="img").to_csv(OUT / "e01_obs.csv", index=False)
    print(f"→ {OUT}：{len(imgs)} 景，{len(o)} 条观测")


if __name__ == "__main__":
    main()
