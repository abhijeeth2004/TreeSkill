# 优化器详解（当前代码）

## 两条优化路径

TreeSkill 当前有两条可执行优化路径：

1. **兼容 APO（重点：Prompt 级）**
   - 实现类：`treeskill.optimizer.APOEngine`
   - 入口：`python -m treeskill.main --optimize`
   - 核心对象：`Skill`（单个 prompt）+ `Trace`
   - 支持：beam search、beam_rounds、tree-aware `evolve_tree`、auto split

2. **主线 ASO（重点：Program 级）**
   - 实现类：`treeskill.aso_optimizer.ASOOptimizer`
   - 入口：`python -m treeskill sealqa-lifecycle` / `python -m treeskill sealqa-aso`
   - 核心对象：`ASOProgram`（root + skills + selection policy）
   - 支持：frontier、auto_prune、auto_merge

## APOEngine（兼容层）

`APOEngine` 的单次优化循环：

1. 从 trace 中取带反馈样本（`t.feedback`）
2. `judge` 角色生成文本梯度
3. `rewrite` 角色生成多候选 prompt 重写
4. `judge` 角色打分候选，并按 `beam_width` / `beam_rounds` 选优

可配置项（`APOConfig`）：
- `beam_width`
- `branch_factor`
- `beam_rounds`
- `num_candidates`
- `gradient_accumulation_steps`

## 树优化（兼容层）

`APOEngine.evolve_tree()` 提供树级扩展：
- `analyze_split_need()`：判断是否需要拆分
- `generate_child_prompts()`：为子技能生成系统提示词
- `evolve_tree()`：自底向上递归优化并支持断点恢复

## ASOOptimizer（主线）

`ASOOptimizer.run()` 的主循环：

1. 在 frontier 中选父程序
2. 用失败 trace 计算 program-level 梯度
3. `propose_actions` 生成结构化候选（add / revise / drop / merge / adjust_selection_policy）
4. 评估并筛选 frontier
5. 可选执行 `auto_merge` / `auto_prune`

## 建议落地方式

若你要做“只调参数”的实验，用 `APOEngine`：
- 与历史 slash 流程兼容（`/bad /rewrite /target /optimize`）

若你要做“程序级演化”实验，用 `ASOOptimizer`：
- 走 `demo/sealqa_*.py`，结合 `search_web_lookup`、`verified_fact_lookup`、`selection_policy`

## 常见问题

- `APOEngine` 与 `ASOOptimizer` 可并存；前者用于兼容交互优化，后者用于主线展示。  
- `rewrite_model` 未配置时会回退到 `judge_model`，`judge_model` 未配置时回退到 actor 模型。
