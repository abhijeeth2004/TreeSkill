# TreeSkill 核心架构

## 目标图景

TreeSkill 已明确主线为：**Kode 前向执行 + ASO 结构进化**，并保留 APO 兼容入口供历史/交互式任务继续使用。

```text
SKILL Program (root + skills + policy)
        │
        ▼
  Kode 前向执行（主模型）
        │
        ▼
  评估 / Trace / 反馈
        │
        ├─ APOEngine（兼容层，prompt 级）
        └─ ASOOptimizer（主线，program 级）
              │
              ▼
      新增技能 / 调整技能 / 合并 / 剪枝
              │
              ▼
      写回 / outputs / ckpt
```

## 入口与职责

- `pipeline_main.py`：主线入口（`python -m treeskill`）
  - `sealqa-lifecycle`（默认）：完整 `root -> generate -> evolve -> prune -> merge`
  - `sealqa-aso`：ASO mini 循环
  - `legacy-chat`：透传到兼容交互入口
- `main.py`：兼容入口（`/bad /rewrite /optimize` 等）
- `demo/demo_sealqa_tree_lifecycle.py`：主线 demo 脚本
- `demo/demo_sealqa_aso.py`：ASO mini demo 脚本

## 数据流

### 主线（推荐）

1. 载入 `ASOProgram`（或 root 组织）
2. Kode 使用技能程序执行任务
3. 持久化 `Trace`
4. `ASOOptimizer` 在 frontier 上提案并评估（文本梯度 + 动作重构）
5. 每轮落盘到 `demo/outputs/...`，可继续下一轮或断点续跑

### 兼容 APO（交互式）

1. 人工 /auto judge 产生 `Trace.feedback`
2. `APOEngine` 按 `beam_width`/`beam_rounds` 优化 `Skill`
3. `SkillTree` 模式下可递归处理叶子到根（`evolve_tree`）

## 核心配置

### LLM 角色模型

- `model`：前向执行（Kode actor）
- `judge_model`：APO/ASO 的失败分析
- `rewrite_model`：APO/ASO 的重写动作生成（默认回退到 `judge_model`）

### 优化参数

- `beam_width`：Beam 宽度
- `branch_factor`：每父节点候选数
- `beam_rounds`：Beam 迭代轮数
- `gradient_accumulation_steps`：每轮抽样反馈数

## 模块映射

- `llm.py`：统一 API 访问层（支持角色覆盖）
- `aso_program.py`：`ASOProgram`、`ASOSkill` 数据结构
- `aso_optimizer.py`：主线 ASO 引擎
- `optimizer.py`：兼容 APO 引擎（含树优化）
- `skill.py`：`SKILL.md` 读写
- `skill_tree.py`：目录化 SkillTree
- `tasks/sealqa.py`：SealQA 任务适配
- `registry.py` + `tools.py`：工具与组件注册
- `checkpoint.py`：兼容层断点恢复

## 兼容关系

- `APOEngine` 与 `ASOOptimizer` 均通过 `LLMClient` 的 `judge/rewrite/actor` 角色协作
- `APOEngine.evolve_tree` 和主线 ASO 的 `auto_split/auto_merge/auto_prune` 是两个层面，可并行理解，但实验主线应优先看 `python -m treeskill`
