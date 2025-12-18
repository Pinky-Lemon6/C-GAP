# Scripts

这里放数据处理脚本（后续接入 AgentErrorBench / ToolBench / Who&When）。

约定：
- 输入统一从 `data/raw/...`
- 中间态写到 `data/intermediate/...`
- 训练 JSONL 输出到 `data/sft/agent_{a,b,c}/...`
- 评测集放到 `data/eval/...`
