# APO 自动提示优化原理

treeskill 有两条优化链路：

- **主线**：`ASO + Kode`，优化对象是完整 skill program（`root + selection policy + 子技能`）
- **兼容层**：`APOEngine`（`treeskill.optimizer`），保留交互式 `/bad`、`/rewrite` 习惯

主线和兼容层共享同一套 `llm.judge_model` / `llm.rewrite_model` 角色抽象，所以文档中的“APO 循环”与“ASO 改造”的术语是互通的。

## 两种优化模式

| 模式 | 优化对象 | 入口 | 适用场景 |
|------|----------|------|----------|
| 交互式 | `SkillTree` 中的单个节点 | `python -m treeskill.main --skill ...` | 人工逐条反馈迭代 |
| 全自动（主线） | `ASOProgram`（skill program） | `python -m treeskill` / `python -m treeskill sealqa-aso` | 数据集驱动闭环 |

## 交互式 APO 流程（兼容）

交互式优化沿用 `legacy-chat`，仍然按反馈→梯度→重写→评分的 APO 思路运转：

1. 用户交互并通过 `/bad`、`/rewrite`、`/target` 记录失败与目标
2. `APOEngine` 读取 `TraceStorage` 失败样本
3. `LLMClient` 的 judge 角色计算文本梯度
4. 采用文本梯度模板 + 重写模板生成候选
5. 打分后保留 `beam_width` 个候选（默认单轨）
6. 版本号递增并写回 `SKILL.md` 与 checkpoint

## ASO 主线（推荐）

主线不再只改单个 prompt，而是改程序级对象：

- `add_skill`
- `revise_skill`
- `drop_skill`
- `merge_skills`
- `adjust_selection_policy`

每轮 ASO 都有一个 **frontier**（候选程序池），对每个候选执行：

- 运行失败 trace 抽取
- 计算程序级文本梯度
- proposal 阶段生成多个 action 集合
- 运行验证器打分
- 取 top-k frontier 进入下一轮

如果配置了 `auto_prune/auto_merge`，会在主循环末尾执行 `prune_program()` / `merge_program()`。

## 核心循环（文本梯度 + Beam Search）

1. 失败样本 -> 失败 trace（TGD 训练信号）
2. 文本梯度（`compute_program_gradient`）
3. 生成候选动作与候选程序（`branch_factor`）
4. 验证打分
5. 按 `frontier_size` 保留前 N 个
6. 重复到 `max_iterations`，得到 `best_program`

当 `beam_width=1` 时退化为单轨；`beam_width>1` 时保留多候选降低局部最优风险。

## 断点续跑

主线 ASO 在每轮结束会持久化阶段产物到 `artifact_dir`（`demo/outputs/...`），兼容层会在 `main` 里按 `ResumeState` 落盘 `.evo_resume.json`。

重启后可从上次 frontier / 最佳版本继续，避免已完成前半程重复计算。

## 和兼容层关系

- 你现在看到的 **主文档**、**推荐 demo** 先走 `python -m treeskill`（主线）
- `python -m treeskill.main` 保留 APO 兼容入口，适合回放旧任务和对比实验
  - `APOEngine` / `ASOOptimizer` 与 `APOEngine` 共用 `llm.judge_model`、`llm.rewrite_model` 配置，但目标对象不同：
  - `APOEngine`：单 prompt（或单节点）
  - `ASOOptimizer`：skill program（根 + policy + skills）
