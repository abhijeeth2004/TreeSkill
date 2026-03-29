# 快速开始（主线）

## 1. 安装

```bash
pip install -e .
```

## 2. 配置

```bash
cp demo/example/config.yaml my-config.yaml
# 填入你的 API key
```

## 3. 跑主线（推荐）

```bash
python -m treeskill
```

默认执行 SealQA lifecycle：  
- 从弱 `root` 开始  
- 通过 `search_web_lookup` 等动作生成技能  
- evolve → prune → merge

结果输出到：

- `demo/outputs/sealqa-tree-lifecycle/summary.json`
- `demo/outputs/sealqa-tree-lifecycle/iteration_*/`

## 4. 运行 ASO mini（最小前沿循环）

```bash
python -m treeskill sealqa-aso
```

## 5. 兼容模式（手工 /dataset 测试）

```bash
python -m treeskill.main --config my-config.yaml --skill demo/example
python -m treeskill.main --config my-config.yaml --skill demo/example --dataset demo/data/paper_cls_train.jsonl --optimize
```

## 6. 常用命令

```bash
python -m treeskill.main --help
python -m treeskill --help
```

常见 slash 命令（交互模式）：
`/bad`、`/rewrite`、`/target`、`/optimize`、`/tree`、`/select`、`/split`、`/ckpt`、`/restore`。
