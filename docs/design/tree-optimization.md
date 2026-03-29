# 树感知优化原理

TreeSkill 的树优化分为两类：  
- **兼容层**：`treeskill.optimizer.APOEngine.evolve_tree()`，按 SkillTree 结构做递归提示词优化  
- **主线 ASO**：`ASOOptimizer` 以 program 为粒度做 `add_skill / revise / drop / merge / adjust_selection_policy`，并可在后处理执行 `auto_prune` / `auto_merge`

## Skill 树结构（当前持久化形式）

```
my-skills/
├── SKILL.md              # 根 skill（通用）
├── social/
│   ├── SKILL.md          # 子 skill
│   ├── moments/
│   │   └── SKILL.md      # 叶子 skill
│   └── weibo/
│       └── SKILL.md      # 叶子 skill
└── business/
    ├── SKILL.md
    ├── email/
    │   └── SKILL.md
    └── product/
        └── SKILL.md
```

每个目录一个 `SKILL.md`，同名子目录就是子技能。

## 兼容层树优化流程（`APOEngine`）

主流程是 **bottom-up**：

1. 从叶子开始优化
2. 再优化父节点（可访问子节点反馈）
3. 根据冲突反馈调用 `analyze_split_need`
4. 有需要时调用 `generate_child_prompts` 自动拆分

### 自动拆分

- 条件：同一个技能的反馈在领域/风格上明显分歧
- 操作：产出 `[{name, description, focus}]`，生成子技能 prompt
- 结果：树中新增子节点，继续递归优化新增节点

### 为什么不是“显式剪枝”

`APOEngine.evolve_tree` 负责**增长与拆分**，不会直接替你删除节点；删并归并主要靠：
- 人工 `/split`/树重构决策（兼容层）
- ASO 后处理的 `auto_prune`（主线 ASO）

## ASO 程序级优化中的 prune / merge

`ASOOptimizer` 在每轮候选评估后可执行后处理：
- `auto_prune`：移除对验证集不增益甚至降分的 skill
- `auto_merge`：将重复/重叠 skill 合并成更小集合

这对应你看到的 demo 阶段：`root -> generate -> evolve -> prune -> merge`。

## 使用示例（兼容层）

```python
from treeskill import GlobalConfig, APOEngine, LLMClient, SkillTree

config = GlobalConfig()
llm = LLMClient(config)
engine = APOEngine(config, llm)

tree = SkillTree.load("my-skills/")
traces = []  # 来自 TraceStorage 或评估数据
engine.evolve_tree(tree, traces, auto_split=True)
tree.save("my-skills-optimized/")
```
