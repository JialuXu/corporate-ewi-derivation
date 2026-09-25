# 对公贷后预警指标自动衍生

[![CI](https://github.com/JialuXu/corporate-ewi-derivation/actions/workflows/ci.yml/badge.svg)](https://github.com/JialuXu/corporate-ewi-derivation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **English summary.** A research framework that auto-derives *interpretable* early-warning
> rules for corporate post-loan credit risk. Atomic metrics carry a type system (L1), typed
> operators enforce legal combinations (L2), a typed genetic-programming search with NSGA-II
> optimises IV / OOT-KS / stability / monotonicity / simplicity / non-redundancy per
> industry × label slice (L3), and an LLM turns each expression tree into a business-language
> card that is then machine-checked against the tree (L4). Everything runs on synthetic data;
> documentation is in Chinese. See [DESIGN.md](DESIGN.md) for the architecture.

面向**对公 · 贷后 · 预警 · 可解释**场景的指标自动衍生框架：
**L1 数据层 + L2 算子库 + L3 GP 多目标搜索 + L4 LLM 解释 + 评估**，
外加 **跨批次台账 + 行业知识 RAG**（`analysis/` + `knowledge/`），
为 L4 之上的元分析（语义聚类 / 冗余识别 / 五类陷阱自检）和单卡可疑度评分提供基础。

从多源原子指标出发自动衍生**候选预警规则指标**，每条衍生指标都能由 LLM 翻译成业务语言。
业务交互层（L5）与真实数据接入不在本仓库范围。架构与设计取舍见 [DESIGN.md](DESIGN.md)。

## 免责声明

- 本项目是研究与工程框架，产出的是**候选**指标，不构成任何信贷决策、投资或合规建议。
- 仓库中的数据全部为合成数据，演示结果不代表任何真实客户、机构或市场的情况。
- 将本框架用于真实业务前，使用方须自行完成数据合规审查、模型验证和人工复核，并对使用结果负责。
- LLM 生成的解释可能有误，即使通过了机器一致性校验，也必须经业务人员复核。

## 数据与本仓库的边界

**本仓库不含任何真实业务数据。** 所有测试与演示跑在 `synthetic.py` 生成的合成数据上。

原子指标清单分两份：

| 清单 | 位置 | 说明 |
|---|---|---|
| 示例清单 | `data/registry/metrics.example.csv`（已提交，76 行） | 仅含公开通用的会计 / 征信 / 工商 / 司法术语，**默认加载**，保证 clone 下来即可运行 |
| 完整清单 | 不在仓库内 | 属于机构内部生产表结构。通过 `AD_REGISTRY_CSV` 指向本地副本，或放到 `data/registry/metrics.csv`（已 gitignore，存在即自动优先加载） |

跑在示例清单上时日志会打一条 `Loaded the example inventory … demo data` 提示，
避免把演示结果误当成真实清单的产出。

合成数据使用示例清单的 `metric_id`。如果本地放了私有清单，跑合成数据演示时请临时指定
`AD_REGISTRY_CSV=data/registry/metrics.example.csv`，否则字段名会被解析到私有清单的编号上，与合成面板对不上。

## 技术栈

- Python ≥ 3.11，包管理 [`uv`](https://github.com/astral-sh/uv)
- Polars（表达式 + 时序窗口）+ DuckDB（跨表 join / 同群分位）+ Parquet
- pydantic v2、scipy、scikit-learn、typer
- ruff + pyright + pytest

## Quickstart

```bash
# 1. 安装依赖
uv sync --extra dev

# 2. 浏览原子指标注册表
uv run derive registry list

# 3. 生成合成数据（500 客户 × 24 月，注入了"营收下滑+司法新增→违约"信号）
uv run derive gen-synthetic

# 4. 评估一条 S-表达式
uv run derive eval \
    --expr '(GT (PctChange 营业收入 12) -0.3)' \
    --label Y_overdue_3m \
    --split oot

# 5. 跨源指标示例：营收恶化 AND 不良信贷余额上升
uv run derive eval \
    --expr '(AND (LT (PctChange 营业收入 12) -0.1) (GT (Delta 未结清不良类表内信贷余额 3) 0))' \
    --label Y_overdue_3m

# 6. 跑 GP 多目标搜索（NSGA-II），输出 Pareto 前沿
uv run derive search --label Y_npl_12m --pop 60 --gens 15 --topk 8

# 7. 调用 LLM 给一棵树生成业务解释卡片（需 AD_LLM_API_KEY；OpenAI-compatible 端点）
uv run derive explain \
    --expr '(GT (PctChange 营业收入 12) -0.1)' \
    --label Y_npl_12m

# 8. 把 GP 结果写入跨批次台账（parquet append-only，Hive partition by industry）
uv run derive search --label Y_npl_12m --industry 制造 --record --seed 1

# 9. 查看台账 + 跨批次全局 Pareto 前沿（Markdown 输出）
uv run derive analysis ledger-list --industry 制造
uv run derive analysis frontier --industry 制造 --out frontier.md

# 10. 验证行业知识库（TF-IDF 字符 n-gram，无分词器）
uv run derive knowledge build
uv run derive knowledge query "应付账款 占款" --industry 批零

# 11. LLM 元分析 + 单卡可疑度评分（需 AD_LLM_API_KEY；输入是 explain 产物的 JSON）
uv run derive analysis report  --cards-glob "explanations/*.json" --industry 制造 --out report.md
uv run derive analysis suspect --cards-glob "explanations/*.json" --industry 制造 --out suspicion.md

# 12. 测试 + 静态检查 + 防泄露检查
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run --extra dev pyright src
python3 scripts/check_public_safety.py
```

贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，尤其是其中的数据边界要求。

## 仓库结构

```
src/auto_derivation/
  l1_data/         # 原子指标注册表、面板 IO、标签构建
  l2_operators/    # 七大类算子（时序变化/异常/事件/同群/逻辑/跨源/关联）
  expression/      # 表达式树、S-表达式解析、类型检查、Polars 求值器
  evaluation/      # 时间切分、IV/KS/Gini、PSI/KL/Wasserstein
  l3_search/       # 类型化 GP（init/variation/fitness/NSGA-II/search/LLM coach）
  l4_explain/      # ExplanationCard、prompt 拼装、LLM 客户端、机器一致性校验
  analysis/        # 跨批次台账 + 全局 Pareto 前沿 + LLM 元分析 + 可疑度评分 + Markdown 渲染
  knowledge/       # 行业知识 RAG（TF-IDF 字符 n-gram）
    corpus/        # 业务专家增量沉淀的 md，已含 制造 / 批零 / 建筑 first-order
  synthetic.py     # 合成数据生成器
  cli.py           # `derive` 入口（registry/gen-synthetic/eval/search/explain/analysis/knowledge）
data/
  registry/
    metrics.example.csv  # 已提交的脱敏示例清单（默认加载）
    metrics.csv          # 可选：私有完整清单，gitignored
  synthetic/       # gen-synthetic 的输出
  analysis_store/  # `derive analysis` 系列的 ledger（parquet，append-only）
tests/             # 单元 + 端到端注入信号回归 + analysis/knowledge
scripts/
  check_public_safety.py  # 防泄露检查（pre-commit + CI）：私有清单 / 凭证 / 内部文档
DESIGN.md          # 设计说明（L1–L5、适应度、五类陷阱）
```

## 表达式语法（S-表达式）

```
expr := atom | "(" OP expr* ")"

OP   ∈ {Delta, PctChange, Mean, Std, ZScore, Slope, TsRank,
        Count, DaysSince,
        PeerRank, PeerDeviation,
        GT, LT, AND, OR, NOT, IfThenElse,
        Ratio, Inconsistency,
        AffilSum, AffilHas (stub)}

atom := 数字字面量 | "字符串" | 字段名（中文 name_cn 或 metric_id）
```

字段名优先按中文 `name_cn` 在 `MetricRegistry` 中查找，未命中则按 `metric_id` 查；解析后 evaluator 会用 `metric_id` 作为 wide-panel 列名。

## 关键设计决策

- 算子 = dataclass + `compile(args, literals, ctx) → pl.Expr`，整棵树**一次编译、一次扫描**
- 物理布局是长表 Parquet；DuckDB `PIVOT` 出宽表后由 Polars 求值
- 类型检查同时执行硬约束（树深 ≤ 5、叶子 ≤ 8、金额/计数 Ratio 黑名单、财务字段窗口约束）
- L3 GP 是手写的**类型化** GP：每次变异都 typecheck，无效返回原 parent；NSGA-II 6 维 fitness `(IV, KS_oot, stability=1/(1+PSI), 单调性, -leaves, -redundancy)`
- L4 LLM 客户端走 OpenAI-compatible 端点（默认 DeepSeek，可换 OpenAI / Moonshot / vLLM）；mock client 用于无 API key 测试；机器一致性检查捕捉字段幻觉、阈值方向反转、统计量漂移
- `analysis/` 与 L1–L4 **零侵入**：通过 wrapper 读 `SearchResult` / `ExplanationResult` 的公开 schema 入账；JSON 解析失败 / cold-start corpus 全部降级处理
- Ledger 用 parquet append-only + Hive partition (`industry=…`)，一个 run 一个文件，并发安全；嵌套字段以 JSON 字符串列存储，schema 完全扁平，便于下游直接消费
- 知识库检索用 sklearn 的 `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))`——免分词、无 embedding 依赖；超过 500 篇或需要近义词查询时再升级

## 已知缺口

- 关联方维度（`AffilSum` / `AffilHas`）需要实体关系图层，目前仅注册签名（示例清单已含 `CASH_0008` / `CASH_0009` 关联方资金往来字段，可作图层落地前的代理信号）
- L3 LLM 教练已有真实实现（`l3_search/llm_coach_real.py`，`derive search --coach real` 启用；seeds / critique / cross_source_proposals 三角色均接入搜索循环），但 prompt 尚未结合 `knowledge` 模块的行业 RAG 检索
- `knowledge/corpus/cross_cutting/` 和 `pitfalls/` 当前为空——前者待沉淀财务舞弊信号 / 资金流红线 / 监管口径等通用知识；后者待补五类陷阱的脱敏案例
- 业务交互层（L5）与上线规则的 drift 检测不在本仓库范围

## 运行环境

- macOS / Linux，Python 3.11–3.13
- 单机 CPU 即可
