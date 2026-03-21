# Evo-Framework Demo

## 准备

```bash
cd /Users/mzm/code/evo_agent
conda activate pr
```

项目根目录的 `.env` 已配置好 SiliconFlow API。

## Demo 1: 从零开始构建

从一句「你是一个写作助手」出发，通过反馈和 APO 自动进化出详细的 prompt：

```bash
python demo/demo_from_scratch.py
```

**流程**: 创建极简 Skill → 生成内容 → 用户反馈（/bad + /rewrite） → 设置 /target 优化方向 → APO 优化 → 对比进化效果

## Demo 2: 从已有 Skill 开始优化

加载预设的写作助手 skill，批量测试多种写作任务，收集反馈后优化：

```bash
python demo/demo_from_skill.py
```

**流程**: 加载 writing-skills.yaml → 多任务生成 → 批量反馈 → 设置 target → APO 优化 → A/B 对比

## 交互式聊天

直接进入 CLI 聊天和实时优化：

```bash
python -m evoskill.main --skill demo/writing-skills
```

主要命令：`/bad`、`/rewrite`、`/export-dpo`、`/target`、`/optimize`、`/image`、`/save`、`/quit`
