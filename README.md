# C-GAP

面向多智能体系统（MAS）运行日志的“因果归因”研究型框架实现（baseline + 可扩展接口）。

本仓库按设计思路实现了一个四阶段流水线：
1) Phase I: Parsing & Compression（Agent A，结构化解析为 I-A-O-T 中间态）
2) Phase II: Sparse Graph Construction（Agent B，稀疏因果图构建：窗口 + 指令栈 + 备用检索）
3) Phase III: Hybrid Pruning（Agent C，PageRank + 语义相似度混合剪枝）
4) Phase IV: Diagnosis（Agent D，黄金上下文格式化 + 根因输出；目前为无外部 LLM 的基线）

当前实现目标：
- 先把工程骨架、数据目录、标准中间态（IAOT）和可运行的 baseline 打通
- 后续你引入 AgentErrorBench / ToolBench / Who&When 数据后，再替换 Agent A/B/C 为 SFT/LoRA 模型推理

## 目录约定（数据集先留空，后续直接放入）

- `data/raw/`：原始数据（各种框架日志/原始文件）
- `data/intermediate/`：清洗后的标准中间态（IAOT/NodeStore）
- `data/sft/agent_a/`：Agent A SFT 数据（Raw -> IAOT）
- `data/sft/agent_b/`：Agent B SFT 数据（pairwise causal）
- `data/sft/agent_c/`：Agent C SFT 数据（trajectory scoring flatten）
- `data/eval/who_when/`：Who&When Benchmark
- `data/processed/`：跑 pipeline 的输出工件（graph/keep_list/diagnosis 等）

## 快速开始（baseline）

在仓库根目录：

1) 安装依赖

- `pip install -r requirements.txt`

### 方式 A：完全不安装

- 运行 demo：`python cgap_run.py demo --out data/processed/demo_artifacts.json`
- 跑测试：`pytest -q`
- 跑自己的日志：`python cgap_run.py run --log path/to/log.txt --error "<your error>" --out data/processed/run_artifacts.json`

### 方式 B：可编辑安装（可选）

2) 以可编辑模式安装本项目（让 `src/` 下的 `cgap` 包可被导入）

- `pip install -e .`

3) 运行内置 demo

- `python -m cgap demo --out data/processed/demo_artifacts.json`
- `python -m cgap demo --out data/processed/demo_artifacts.json`

4) 跑单元测试

- `pytest -q`

5) 运行自己的日志

- `python -m cgap run --log path/to/log.txt --error "<your error>" --out data/processed/run_artifacts.json`
- `python -m cgap run --log path/to/log.txt --error "<your error>" --out data/processed/run_artifacts.json`

## 实现说明

- 标准数据结构在 `src/cgap/schema.py`：`StandardLogItem`（含 I/A/O/T + 元信息）和 `NodeStore`
- 四阶段 baseline 在 `src/cgap/agents.py`，统一由 `src/cgap/pipeline.py` 组织
- CLI 入口在 `src/cgap/cli.py`
