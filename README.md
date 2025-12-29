# C-GAP
C-GAP（Causal Graph-based Agent Pipeline）是一个用于**多智能体系统失败归因（root cause attribution）**的 4 阶段流水线。

## 核心架构：Atomic Node-based Causal Graph

本项目采用细粒度的"原子节点"（Atomic Node）架构，每个日志步骤会被拆分为多个语义原子事件，从而构建更精确的因果依赖图。

### 原子节点类型

| 类型 | 含义 | 示例 |
|------|------|------|
| **INTENT** | 内部思考、计划、目标、决策 | "I need to search for weather data" |
| **EXEC** | 工具调用、代码执行、API 请求 | "Calling search_api('weather')" |
| **INFO** | 观察结果、错误反馈、系统消息 | "API returned: temperature=25°C" |
| **COMM** | 与其他 Agent 或用户的通信 | "Forwarding request to Agent B" |

### 四阶段流水线

1. **Phase I（Atomic Extraction）**：使用 LLM 将原始日志拆分为原子节点列表（INTENT/EXEC/INFO/COMM），支持并行处理。
2. **Phase II（Causal Graph Building）**：两阶段构图——Step 内硬规则串联（Intra-Step）+ Step 间 LLM 验证连边（Inter-Step），输出带类型标注的有向因果图。
3. **Phase III（Deterministic Slicing）**：基于加权反向遍历的确定性图切片算法（无 LLM），包含语义过滤与循环压缩，高效提取与失败相关的因果链。
4. **Phase IV（Root Cause Diagnosis）**：将切片后的因果图转为 Golden Context（含边注释、Gap Summary、Error 高亮），调用 LLM 输出结构化诊断结果。

本仓库默认以"OpenAI 兼容接口"的方式调用模型。

---

## 目录结构

```
C-GAP/
├── main.py                      # 命令行入口，串联 Phase I–IV
├── src/
│   ├── llm_client.py            # OpenAI SDK(v1) 兼容封装
│   ├── models.py                # 数据结构（AtomicNode, StandardLogItem, TaskContext 等）
│   ├── utils.py                 # 数据加载、归一化、中间结果保存
│   └── pipeline/
│       ├── causal_types.py      # NodeType/CausalType 枚举与类型约束
│       ├── candidate_selector.py# 候选节点选择（规则 + 语义双轨）
│       ├── phase1_parser.py     # Phase I: 原子节点抽取
│       ├── phase2_builder.py    # Phase II: 因果图构建
│       ├── phase3_pruner.py     # Phase III: 确定性图切片
│       └── phase4_diagnoser.py  # Phase IV: 根因诊断
├── config/                      # 配置文件目录
└── data/
    ├── raw/                     # 原始输入数据
    ├── intermediate/            # 各阶段中间产物
    └── processed/               # 处理后的数据
```

---

## 环境与依赖

建议 Python 3.10+。

**核心依赖**：
- `openai` - LLM API 调用
- `pydantic` - 数据模型验证
- `networkx` - 图结构与算法
- `python-dotenv` - 环境变量管理
- `numpy` - 数值计算（用于 Embedding 相似度）

安装依赖（任选其一）：

```bash
# 使用 pip
pip install openai pydantic networkx python-dotenv numpy

# 使用 conda
conda create -n cgap python=3.10 -y
conda activate cgap
pip install openai pydantic networkx python-dotenv numpy
```

---

## 配置模型（.env）

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.example.com/v1
```

> **注意**：`OPENAI_BASE_URL` 必须是 API 根路径，通常以 `/v1` 结尾。代码会自动裁剪误写的 `.../v1/chat/completions` 后缀。

---

## 运行方式

### 基本命令

```bash
python main.py --input data/raw/your_case.json --dataset-type hand_crafted --model deepseek-chat
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/raw/sample_session.json` | 输入文件路径（`.json` 或 `.jsonl`） |
| `--model` | `gpt-4` | 模型名称（取决于服务商） |
| `--dataset-type` | `hand_crafted` | 数据集类型：`hand_crafted` 或 `algorithm` |
| `--top-k` | `30` | Phase III 保留的关键步骤数 |
| `--window-size` | `15` | Phase II 候选选择窗口大小 |
| `--phase1-batch-size` | `16` | Phase I 并行批大小 |
| `--phase1-max-workers` | `16` | Phase I 并发线程数 |
| `--phase1-max-chars` | `100000` | Phase I 单条日志最大字符数 |
| `--use-embeddings` | `True` | Phase II 是否启用 Embedding 候选选择 |

### 输出

- **终端**：打印 Phase IV 的诊断 JSON
- **中间产物**：写入 `data/intermediate/{session_id}/`
  - `phase1_atomic_*.json` - 原子节点抽取结果
  - `phase2_graph_*.json` - 因果图（节点 + 边 + 统计）
  - `phase3_sliced_*.json` - 切片后的节点列表
  - `phase4_diagnosis_*.json` - 最终诊断结果

---

## 输入格式

### A) Who&When 格式（推荐）

`--dataset-type hand_crafted` 与 `--dataset-type algorithm` 都支持这种结构。

**Hand-Crafted 示例**：

```json
{
  "question_ID": "demo_001",
  "question": "用户的目标是什么？",
  "ground_truth": "正确答案是什么（用于诊断对齐）",
  "history": [
    {"role": "User", "content": "请帮我……"},
    {"role": "Orchestrator", "content": "先做A再做B"},
    {"role": "WebSurfer", "content": "打开网页..."}
  ]
}
```

**Algorithm-Generated 示例**（额外带 `name` 字段）：

```json
{
  "question_ID": "demo_002",
  "question": "用户的目标是什么？",
  "ground_truth": "正确答案是什么",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "name": "WebSurfer", "content": "..."}
  ]
}
```

### B) 通用 JSON 格式

```json
{
  "dataset_source": "sample",
  "session_id": "session_X",
  "question": "...",
  "ground_truth": "...",
  "error_info": "...",
  "steps": [
    {"step_id": 0, "role": "user", "raw_content": "..."},
    {"step_id": 1, "role": "assistant", "raw_content": "..."}
  ]
}
```

### C) JSONL 格式

```jsonl
{"step_id": 0, "role": "user", "raw_content": "..."}
{"step_id": 1, "role": "assistant", "raw_content": "..."}
```

---

## 技术细节

### Phase II: 因果图构建

采用双轨候选选择策略：
- **规则轨**：基于节点类型约束（`VALID_CAUSAL_SOURCES`）筛选合法因果源
- **语义轨**：基于 Embedding 相似度选择语义相关候选

因果类型标注：
- `INSTRUCTION`：指令依赖（Source 发出指令，Target 执行）
- `DATA`：数据依赖（Target 使用 Source 产生的信息）
- `STATE`：状态依赖（Target 依赖 Source 建立的状态）

### Phase III: 确定性图切片

核心算法：
1. **加权反向遍历**：从目标节点出发，基于边类型权重的优先队列 BFS
2. **语义过滤**：保留 INTENT、错误相关、高度节点；丢弃孤立 EXEC/死端 INFO
3. **循环压缩**：检测重复模式并折叠为压缩节点

边权重设计：
- `PRIMARY`（主链）: 0.1
- `SECONDARY`（关联）: 1.0
- `FALLBACK`（时序保底）: 5.0

### Phase IV: Golden Context 生成

结构化 Prompt 包含：
- 任务上下文（Question / Ground Truth / Error Info）
- 线性化节点序列（按 step_id 排序）
- 边注释（`[Implicit Context]` / `[WEAK LINK]`）
- 缺失步骤摘要（Gap Summary）
- 错误标记（`[ERROR]` 前缀）

---

## License

MIT License
