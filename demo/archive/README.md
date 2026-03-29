# Archived Demos

这些 demo 保留下来仅作历史参考，不再代表仓库当前主流 pipeline。  
保留原因：保留实验轨迹、排查回归时复现历史行为。  
若要在当前主线继续对照，请优先使用根目录 `demo/` 下的两个入口。

> 注意：归档脚本可能依赖早期配置习惯（包括旧模型名、参数写法、评分策略），不应作为产品默认标准。

## 目录

| 目录 | 内容 |
|---|---|
| `legacy-apo/` | 旧 `APOEngine`、prompt-only、早期 Kode/SealQA/RepoOps 实验 |
| `legacy-tree/` | 早期 tree-aware / showcase / quick experiment 脚本 |

## 当前推荐

请优先使用：

```bash
python -m treeskill
python -m treeskill sealqa-aso
```

对应主线是：
- `Kode` 负责前向
- `ASO` 负责 skill/program 修改
- `SealQA lifecycle` 负责展示 `root -> generate -> evolve -> prune -> merge`
