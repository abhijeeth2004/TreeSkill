# Demo Guide

当前推荐只看两条主线：

| Demo | 用途 | 命令 |
|---|---|---|
| `demo_sealqa_tree_lifecycle.py` | 完整生命周期：`root -> generate -> evolve -> prune -> merge` | `python -m treeskill` |
| `demo_sealqa_aso.py` | 更接近真实 frontier/beam 的最小 ASO 实验 | `python -m treeskill sealqa-aso` |
| `run_sealqa_demo.sh` | 一键复现入口（支持 `--lifecycle` / `--aso` / `--both`） | `./run_sealqa_demo.sh` |

## 推荐顺序

### 1. 生命周期 Demo

这是当前仓库的主流 pipeline：
- `Kode` 做前向执行
- `ASO` 做 skill/program 修改
- 使用本地 `search_web/fetch_url` 抽象稳定复现 SealQA 小样本
- 检索策略默认：`search_web_lookup` 只查本地缓存；可通过设置 `SEALQA_ASO_ENABLE_WEB_FALLBACK=1` 打开外部回退，并通过 `SEALQA_WEB_SEARCH_CMD`、`SEALQA_WEB_FETCH_CMD` 注入外部工具命令。
 - 示例（Claude Code CLI）：

```bash
export SEALQA_ASO_ENABLE_WEB_FALLBACK=1
export SEALQA_WEB_SEARCH_CMD='claude -p "Use web search only. Return only a JSON array with fields id,title,url,snippet for the query: {query}. Top: {top_k}." --allowed-tools WebSearch --output-format text'
export SEALQA_WEB_FETCH_CMD='claude -p "Fetch {url} and return only the page text excerpt (no markdown)." --allowed-tools WebFetch --output-format text'
```

说明：

- 命令里的 `{query}`、`{top_k}`、`{url}` 会由脚本按样本问题自动替换。
- `--allowed-tools` 仅用于 Claude Code 本地命令行环境，失败时会自动回退到当前 cache。

```bash
python -m treeskill
```

输出目录：

```text
demo/outputs/sealqa-tree-lifecycle/
```

### 2. 最小 ASO Demo

如果你要看 frontier / candidate growth：

```bash
python -m treeskill sealqa-aso
```

输出目录：

```text
demo/outputs/sealqa-aso-mini/
```

### 3. 一键复现（推荐）

```bash
chmod +x demo/run_sealqa_demo.sh
./demo/run_sealqa_demo.sh            # 只跑生命周期主线
./demo/run_sealqa_demo.sh --aso      # 只跑 ASO 简化链路
./demo/run_sealqa_demo.sh --both     # 连续跑两条主线
```

默认并发和模型：
```bash
export KODE_ACTOR_MODEL="${KODE_ACTOR_MODEL:-MiniMax-M2.7}"
export KODE_ACTOR_PROTOCOL="${KODE_ACTOR_PROTOCOL:-anthropic}"
export KODE_ACTOR_BASE_URL="${KODE_ACTOR_BASE_URL:-https://api.minimaxi.com/anthropic}"
export SEALQA_EVAL_MAX_WORKERS="${SEALQA_EVAL_MAX_WORKERS:-8}"
export SEALQA_ASO_MAX_WORKERS="${SEALQA_ASO_MAX_WORKERS:-8}"
export MINIMAX_API_KEY="your_minimax_key_here"
export TREE_LLM_JUDGE_PROTOCOL="${TREE_LLM_JUDGE_PROTOCOL:-anthropic}"
export TREE_LLM_JUDGE_MODEL="${TREE_LLM_JUDGE_MODEL:-MiniMax-M2.7}"
export TREE_LLM_REWRITE_MODEL="${TREE_LLM_REWRITE_MODEL:-MiniMax-M2.7}"
```

## 归档 Demo

旧的 prompt-only / `APOEngine` / 早期 tree 实验，已经迁到：

```text
demo/archive/
```

它们保留作历史参考，不再代表当前主流 pipeline。
