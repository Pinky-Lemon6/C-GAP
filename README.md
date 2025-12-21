# C-GAP

C-GAP 是一个用于“多智能体系统失败归因（root cause attribution）”的 4 阶段流水线：

1. Phase I（Parser / Agent A）：把原始日志清洗为结构化 I-A-O-T（Instruction / Action / Observation / Thought）。
2. Phase II（Builder / Agent B）：用“指令栈 + 滑动窗口 + LLM 验证”构建因果依赖图。
3. Phase III（Pruner / Agent C）：用 PageRank + LLM 语义相关性打分，挑选 Top-K 关键步骤作为 Golden Context。
4. Phase IV（Diagnoser / Agent D）：基于 Golden Context（含因果引用 tag）、question / ground_truth / error_info，输出根因步骤与责任角色。

本仓库默认以“OpenAI 兼容接口”的方式调用模型，并强制各阶段输出 JSON（通过 `response_format={"type":"json_object"}`）。

---

## 目录结构

- `main.py`：命令行入口，串起 Phase I–IV，并把中间产物落盘到 `data/intermediate/`。
- `src/llm_client.py`：OpenAI SDK(v1) 兼容封装，支持 `.env`，并自动修正常见 `base_url` 配置错误。
- `src/models.py`：数据结构（`StandardLogItem`、`IAOT` 等）。
- `src/utils.py`：数据加载、role 归一化、error_info 构造、jsonl 读写、中间结果保存。
- `src/pipeline/phase1_parser.py` ~ `phase4_diagnoser.py`：四阶段实现。

---

## 环境与依赖

建议 Python 3.10+。

安装依赖（任选其一）：

- 使用 pip：
  - `pip install openai pydantic networkx python-dotenv`
- 使用 conda（示例）：
  - `conda create -n cgap python=3.10 -y`
  - `conda activate cgap`
  - `pip install openai pydantic networkx python-dotenv`

---

## 配置模型（.env）

在项目根目录创建/修改 `.env`（如果你已经有就直接改）：

- `OPENAI_API_KEY=你的key`
- `OPENAI_BASE_URL=https://xxx/v1`

注意：
- `OPENAI_BASE_URL` 必须是“API 根路径”，通常以 `/v1` 结尾。
- 如果你误写成 `.../v1/chat/completions`，代码会做一次自动裁剪归一化，但仍建议你改成正确值。

---

## 运行方式

最常用命令：

- `python main.py --input data/raw/your_case.json --dataset-type hand_crafted --model your-model-name`

常用参数：

- `--input`：输入文件路径（`.json` 或 `.jsonl`）。
- `--dataset-type`：`hand_crafted` 或 `algorithm`。
- `--model`：模型名（取决于你的服务商/网关）。
- `--top-k`：Phase III 保留的关键步骤数。
- `--window-k`：Phase II 的滑动窗口大小。

输出：
- 终端打印 Phase IV 的诊断 JSON。
- 中间产物写入 `data/intermediate/{session_id}/`，包括：
  - `phase1_*.json`：结构化步骤（I-A-O-T）
  - `phase2_graph_*.json`：图边列表
  - `phase3_pruned_*.json`：保留的 step_id
  - `final_result_*.json`：最终诊断

---

## 输入格式

### A) Who&When（推荐）：JSON 对象 + `history`

`--dataset-type hand_crafted` 与 `--dataset-type algorithm` 都支持这种结构。

最小示例（hand-crafted）：

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

algorithm-generated（额外带 `name` 字段）：

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

实现细节：
- `role` 会被 `normalize_role()` 归一化为小写并去掉括号后缀。
- algorithm 格式会把 `name` 注入到 `raw_content`（如 `SPEAKER_NAME: WebSurfer`），以便 Phase I 更好抽取 I-A-O-T。

### B) 通用 JSON：steps 数组

适合你自己构造/调试：

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

### C) JSONL：一行一个 step 对象

```json
{"step_id": 0, "role": "user", "raw_content": "..."}
{"step_id": 1, "role": "assistant", "raw_content": "..."}
```

