# 贡献指南

欢迎提 Issue 和 PR。设计背景见 [DESIGN.md](DESIGN.md)，工程用法见 [README](README.md)。

## 开发环境

```bash
uv sync --extra dev
uv tool install pre-commit && pre-commit install   # 提交前自动运行防泄露检查和 ruff
```

提交前请确保以下命令全部通过（CI 会运行同样的检查）：

```bash
python3 scripts/check_public_safety.py
uv run --extra dev ruff check src tests scripts
uv run --extra dev pyright src
uv run --extra dev pytest
```

## 数据边界（最重要）

本仓库是公开的，**任何真实业务数据都不能进入仓库**，包括代码、测试、文档、Issue、PR 描述和提交说明。

- 测试和示例只使用 `synthetic.py` 生成的合成数据。
- 不要提交机构内部的指标清单、表结构、字段名或其统计信息（行数、各来源分布等）。
  使用自己的清单时，通过 `AD_REGISTRY_CSV` 指向本地文件，或放到 `data/registry/metrics.csv`（已 gitignore）。
- 如果本地有私有清单，`scripts/check_public_safety.py` 会自动检查其中的字段名有没有出现在待提交的文件里。
  还可以把其他不应公开的词写进本地的 `.public-safety-denylist`（已 gitignore，每行一个）。
- 知识库语料（`src/auto_derivation/knowledge/corpus/`）只写行业通用知识；案例必须完全脱敏。

## 扩展示例指标清单

`data/registry/metrics.example.csv` 只能使用公开标准的会计、征信、工商、司法术语：

- 不使用核心系统厂商的字段名，不包含内部流程或审批层级字段。
- 不使用 PII 类字段名（如证件号码、地址、账号）。
- 文件编码为带 BOM 的 UTF-8；读取时用 `csv` 模块和 `utf-8-sig`（字段值里含逗号）。
- `metric_id` 按来源前缀从 `_0001` 起连续编号，新行接在同前缀最后一行之后（有测试守护）。
- 9 列：`metric_id, source, name_cn, dtype, time_grain, update_freq, value_range, business_tag, null_semantics`。
- `dtype` ∈ {金额, 计数, 天数, 比率, 分类, 布尔, 文本, 日期}；`time_grain` ∈ {日, 季, 年, 事件型}。
- `null_semantics` 要区分"无记录"与"查询失败"等不同的空值含义（DESIGN.md §8 陷阱 2）。
- `synthetic.py` 的 `METRICS` 中用到的 `metric_id` 必须保留在示例清单中（有测试守护）。

## 设计约束

提交的改动需要符合 DESIGN.md 中的约束，尤其是：

- 只面向对公、贷后、预警、可解释场景（DESIGN.md §1）。
- 搜索必须按行业 × 标签分片。
- 不要重新引入 DESIGN.md §8 列出的五类陷阱。
- 数据完整性检查（如面板重复键、不支持的数据类型、时间切分样本不足）保持显式报错。

## 提交规范

- 一个 PR 聚焦一件事；附上测试。
- 提交说明写清楚"改了什么、为什么"，不要引用任何真实数据。
