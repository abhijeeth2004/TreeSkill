# 跨模型 Skill 迁移

TreeSkill 当前支持把**优化角色**和**执行角色**解耦，满足“小模型执行、强模型改写”场景。

## 适用场景

- 主模型（Executor）用于真实前向执行，如 Kode 中的主对话模型
- Judge/Rewrite 模型负责 `APOEngine` / `ASOOptimizer` 的诊断与改写
- 典型目标：用较便宜模型降低成本，或给小模型做更显式的 Skill 生成

`GlobalConfig.llm` 已内置分角色字段：

- `model` / `base_url` / `protocol`：前向执行（actor）
- `judge_model` / `judge_api_key` / `judge_base_url` / `judge_protocol`：失败分析（Judge）
- `rewrite_model` / `rewrite_api_key` / `rewrite_base_url` / `rewrite_protocol`：改写提示词（Rewrite）

`APOEngine`/`ASOOptimizer` 对应用到：

- 失败归因与文本梯度：`role="judge"`（走 `judge_model`）
- 候选重写与 merge/split 动作生成：`role="rewrite"`（走 `rewrite_model`，默认回退到 `judge_model`）
- 验证与前向评分：前向执行模型

## 配置示例

```yaml
llm:
  api_key: "sk-xxx"
  base_url: "https://api.openai.com/v1"
  model: "qwen-plus"                 # 执行模型
  judge_model: "claude-4"            # Judge 角色
  rewrite_model: "claude-4"          # Rewrite 角色（可与 judge_model 共用）
  judge_protocol: "anthropic"
  rewrite_protocol: "anthropic"
```

```bash
export TREE_LLM_MODEL=qwen-plus
export TREE_LLM_JUDGE_MODEL=claude-4
export TREE_LLM_REWRITE_MODEL=claude-4
```

## 实施建议

1. 先用执行模型跑 `--dataset` 生成失败 trace
2. 用更强模型作为 judge/rewrite 角色做优化循环
3. 回到执行模型验证新 skill 的真实收益（accuracy / pass@k / DPO 可用指标）
4. 按任务确定是否保留双模型或退化为单模型（只配 actor）

> 重点：当 `rewrite_model` 不设置时，系统会自动回退到 `judge_model`；`judge_model` 未设置则回退到 `model`，行为和历史配置兼容。
