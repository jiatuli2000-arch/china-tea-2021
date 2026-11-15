# S1 探针点

**目的**：在四大茶区代表区目视解译探针点，作为 TI 构建与检验的依据。

- 输入：`docs/00` 标注协议、`docs/02` 代表区
- 输出：`data/raw/probes/<zone>.csv`（字段见标注协议）
- 验收：`check.py`——字段完整、坐标在代表区内、每茶区每主要类别 ≥ 50 个置信度 ≥ 2 的点、`point_id` 唯一

## 操作步骤

```bash
conda activate tea2021
python stages/S1_probes/make_kit_<area>.py          # 生成打点包 data/interim/S1/<area>_kit_v1.kml
# → Google Earth 网页版标注，导出 KML（流程见 docs/04）
python stages/S1_probes/ingest.py 导出.kml --zone jiangnan --area huangshan --province 安徽 --annotator <你>
python stages/S1_probes/check.py                     # 验收
```

| 代表区 | 打点包 | 状态 |
|---|---|---|
| 黄山（江南） | `huangshan_kit_v1.kml`：旧判读点 241 个 + ESA 2021 分层补充点 145 个（树木 90、农田 40、草地 15；灌丛类在黄山没有抽到像元） | 待标注 |
