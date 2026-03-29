# Core Abstraction Layer 使用指南（v0.2.0）

`treeskill.core` 是框架的模型无关底层抽象层，核心目标是让优化循环与具体模型 API 解耦。

## 抽象分层

- `OptimizablePrompt`：可优化对象（`TextPrompt`、`MultimodalPrompt`、`StructuredPrompt`）
- `TextualGradient`：失败归因到可执行动作的自然语言梯度
- `Experience`/`Feedback`：一次交互 + 反馈样本
- `ModelAdapter`：统一的模型 API 抽象（生成、失败归因、梯度改写、验证）
- `TrainFreeOptimizer`：兼容 APO 的单 Prompt 级 TGD 循环
- `TreeAwareOptimizer`：树结构环境下的兼容层优化

这些接口都在 `treeskill/core/` 下集中实现，`treeskill/__init__.py` 会将常用符号导出为顶层 API。

## 关键抽象与示例

### 1. OptimizablePrompt

```python
from treeskill.core import TextPrompt

prompt = TextPrompt(
    content="你是一个写作助手。",
    name="writing-assistant",
    version="v1.0",
    target="更像人类",
)

serialized = prompt.serialize()
roundtrip = TextPrompt.deserialize(serialized)
next_ver = prompt.bump_version()  # v1.0 -> v1.1
```

### 2. TextualGradient

```python
from treeskill.core import SimpleGradient

grad = SimpleGradient(
    text="提示词过于正式，请改成更口语化。",
    metadata={"source": "user_feedback", "sample_count": 5},
)

str(grad)
```

### 3. Experience / Feedback

```python
from treeskill.core import ConversationExperience, CompositeFeedback

trace_like = ConversationExperience(
    messages=[{"role": "user", "content": "写一首春天的诗"}],
    response="春眠不觉晓...",
)

feedback = CompositeFeedback(score=0.3, critique="太老套了")
with_feedback = trace_like.attach_feedback(feedback)

print(with_feedback.is_failure)  # True
```

### 4. ModelAdapter

```python
from treeskill.core import BaseModelAdapter


class MyModelAdapter(BaseModelAdapter):
    @property
    def model_name(self):
        return "my-model"

    @property
    def supports_vision(self):
        return False

    @property
    def max_context_tokens(self):
        return 8192

    def generate(self, prompt, context=None, temperature=0.7, **kwargs):
        messages = [
            {"role": "system", "content": prompt.to_model_input()},
            *(context or []),
        ]
        return self._call_api(messages=messages, temperature=temperature, **kwargs)

    def _call_api(self, messages, system=None, temperature=0.7, **kwargs):
        # 在这里调用你的模型 SDK / HTTP API
        raise NotImplementedError

    def _count_tokens_impl(self, text: str):
        return len(text.split())
```

`BaseModelAdapter` 已内置 `compute_gradient` / `apply_gradient` / `validate_prompt` 的默认实现；
具体模型只需要实现上面三个底层方法即可。

## 训练自由优化工作流示例（兼容层）

```python
from treeskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    TrainFreeOptimizer,
    OptimizerConfig,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
)

prompt = TextPrompt(content="你是一个助手。", name="base")
adapter = MyModelAdapter(model_name="demo-model")
samples = [
    ConversationExperience(
        messages=[{"role": "user", "content": "写个故事"}],
        response="一次性输出...",
        feedback=CompositeFeedback(score=0.2, critique="太呆板"),
    )
]
config = OptimizerConfig(max_steps=2, gradient_accumulation_steps=5)
base_optimizer = TrainFreeOptimizer(adapter=adapter, config=config)
tree_optimizer = TreeAwareOptimizer(
    adapter=adapter,
    base_optimizer=base_optimizer,
    config=TreeOptimizerConfig(),
)

optimized = base_optimizer.optimize(prompt=prompt, experiences=samples, validator=None)
```

## 多模态示例

```python
from treeskill.core import MultimodalPrompt

prompt = MultimodalPrompt(
    text="这张图里有什么？",
    images=["/tmp/photo.jpg"],
    name="vision-demo",
)
```

`BaseModelAdapter.validate_prompt()` 会检查 token 上限、vision 能力兼容性与接口完整性。

## 序列化与持久化

```python
from pathlib import Path
import json

Path("prompt.json").write_text(json.dumps(prompt.serialize(), ensure_ascii=False, indent=2))
prompt2 = TextPrompt.deserialize(json.loads(Path("prompt.json").read_text()))

Path("trace.jsonl").write_text(json.dumps(with_feedback.to_training_sample()) + "\n")
```

## 扩展建议

### 新增 Prompt 类型

需要同时实现 `OptimizablePrompt` 的契约：`to_model_input / serialize / deserialize / apply_gradient / bump_version`。

### 新增 Feedback 类型

需要实现 `to_score / to_dict / is_negative` 并提供与 `ConversationExperience`/`MultimodalExperience` 兼容的数据提取逻辑。

## 常见问题

**Q: 我可以流式返回 token 吗？**  
A: 当前底层 `ModelAdapter.generate()` 返回普通文本；若要流式，可在自定义适配器内返回可迭代对象并在上层封装。

**Q: 为什么要两个优化器？**  
A: `TrainFreeOptimizer` 和 `TreeAwareOptimizer` 是 prompt / tree 兼容层能力，当前主线 ASO 优化走 `ASOOptimizer`（位于 `treeskill.aso_optimizer`），目标是 skill program 级别演化。

## 参考

- [Textual Gradient Descent 论文](https://arxiv.org/pdf/2502.16923)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [OpenAI API 文档](https://platform.openai.com/docs/)
