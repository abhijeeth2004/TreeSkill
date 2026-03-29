# TreeSkill 使用指南

> 版本：v0.2.0（与当前主线对齐）  
> 更新时间：2026-03-30

---

## 1. 配置系统

TreeSkill 支持三层配置：环境变量、`.env`、`--config` YAML，优先级是环境变量 > `.env` > YAML > 默认值。

### 推荐的最小配置

```yaml
llm:
  api_key: "sk-xxx"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"           # actor
  judge_model: "gpt-4o"          # APO/ASO judge
  rewrite_model: "gpt-4o"        # APO/ASO rewrite
  protocol: "openai"

storage:
  trace_path: "./data/traces.jsonl"
  skill_path: "./skills"

apo:
  beam_width: 1
  branch_factor: 2
  beam_rounds: 3
  gradient_accumulation_steps: 5
  num_candidates: 2

verbose: true
```

### 1.1 环境变量

```bash
export TREE_LLM_API_KEY="sk-xxx"
export TREE_LLM_BASE_URL="https://api.openai.com/v1"
export TREE_LLM_MODEL="gpt-4o-mini"
export TREE_LLM_JUDGE_MODEL="gpt-4o"
export TREE_LLM_REWRITE_MODEL="gpt-4o"
```

### 1.2 使用 YAML 加载

```bash
python -m treeskill --help
python -m treeskill.main --config demo/example/config.yaml --skill demo/example
```

---

## 2. Skill 文件（Agent Skills 格式）

TreeSkill 当前主线和兼容层都使用目录化 `SKILL.md`。

```text
my-skill/
├── SKILL.md       # 必须，frontmatter + Markdown system prompt
├── config.yaml    # 可选，skill 级参数
└── script.py      # 可选，工具函数
```

`SKILL.md` 示例：

```markdown
---
name: my-writing-assistant
description: 专业中文写作助手
metadata:
  version: "1.0"
  target: "更像真人说话，有温度"
---

你是一位专业的中文写作助手。
语言要自然，避免 AI 腔，给出可落地的写作建议。
```

对应字段：
- `metadata.version` → `Skill.version`
- `metadata.target` → `Skill.target`
- 内容正文 → `Skill.system_prompt`

---

## 3. 快速开始（主推荐）

### 运行主线 demo（Kode + ASO）

```bash
# 主线：SealQA lifecycle（默认）
python -m treeskill

# 仅 ASO mini
python -m treeskill sealqa-aso

# 兼容模式
python -m treeskill legacy-chat
python -m treeskill.main --skill demo/example --ckpt ckpt/example-run
```

主线输出保存在 `demo/outputs/`，可查看 `summary.json`、各迭代 `iteration_*`。

### 手工加载与交互（兼容）

```bash
python -m treeskill.main --config demo/example/config.yaml --skill demo/example
```

常见 slash 命令：`/bad`、`/rewrite`、`/target`、`/optimize`、`/tree`、`/split`、`/ckpt`、`/restore`、`/quit`。

---

## 4. 数据集驱动优化

### 自动评估 + APO

```bash
python -m treeskill.main --config demo/example/config.yaml \
  --skill demo/example --dataset demo/data/sealqa_tree_samples.json \
  --optimize
```

### 人工标注（HITL）

```bash
python -m treeskill.main --optimize --annotate --dataset demo/data/sealqa_tree_samples.json
```

### 注意

- `--manual`：强制人类 judge
- `--no-resume`：跳过断点恢复提示，直接重跑

---

## 5. 工具系统

SKILL 的工具来自三类：
- `script.py`：本地 Python 工具
- `config.yaml` 中定义的 HTTP/MCP 工具
- `~/.kode` / MCP 外部接入（运行时按需）

运行时配置可在 `demo/example/config.yaml` 与 `demo/example/SKILL.md` 查看。

---

## 6. 兼容说明

- `python -m treeskill`：当前主线（推荐）
- `python -m treeskill.main`：兼容 AP O / 交互式流程
- `demo/archive/`：历史实验与旧思路（保留但不作为主线）

---

## 7. 相关文档

- [APO 优化原理](./design/apo-optimization.md)
- [树感知优化](./design/tree-optimization.md)
- [实验记录](./EXPERIMENTS.md)
- [Skill 文件格式](./design/skill-format.md)
