"""Mock adapter for testing and demonstration.

This adapter doesn't call any real API, but demonstrates how to
implement the BaseModelAdapter interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evoskill.core.base_adapter import BaseModelAdapter
from evoskill.core.abc import OptimizablePrompt, Experience


class MockAdapter(BaseModelAdapter):
    """A mock adapter for testing purposes.

    This adapter simulates responses without calling any API.
    Useful for unit tests and demonstrations.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="mock-model", **kwargs)
        self.call_count = 0
        self.last_messages = None

    @property
    def supports_vision(self) -> bool:
        return True  # Mock supports everything

    @property
    def max_context_tokens(self) -> int:
        return 100000

    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a mock response."""
        prompt_text = self._extract_prompt_text(prompt)

        # Simulate different responses based on prompt content
        if "写作" in prompt_text or "write" in prompt_text.lower():
            return "这是一个模拟的写作回复。"
        elif "代码" in prompt_text or "code" in prompt_text.lower():
            return "def example():\n    return 'mock code'"
        else:
            return f"Mock response to: {prompt_text[:50]}..."

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Simulate API call."""
        self.call_count += 1
        self.last_messages = messages

        messages_str = str(messages)

        # Simulate gradient computation (analysis)
        if "分析" in messages_str and ("failures" in messages_str.lower() or "失败" in messages_str):
            return (
                "分析结果显示，当前Prompt存在以下问题：\n"
                "1. 指令不够具体\n"
                "2. 缺少示例\n"
                "3. 语气过于正式\n\n"
                "建议修改这些方面。"
            )

        # Simulate prompt rewrite
        if "rewrite" in messages_str.lower() or "重写" in messages_str:
            return (
                "你是一个专业且友好的助手。\n\n"
                "你的任务是：\n"
                "1. 理解用户需求\n"
                "2. 提供清晰、准确的回答\n"
                "3. 保持友善和专业的语调\n\n"
                "记住：简洁胜于冗长。"
            )

        # Default response
        return "Mock API response"

    def _count_tokens_impl(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4


# Example usage
if __name__ == "__main__":
    from evoskill.core import (
        TextPrompt,
        ConversationExperience,
        CompositeFeedback,
    )

    print("=" * 60)
    print("MockAdapter Demo")
    print("=" * 60)

    # 1. Create adapter
    adapter = MockAdapter()
    print(f"\n✓ Created adapter: {adapter.model_name}")
    print(f"  - Supports vision: {adapter.supports_vision}")
    print(f"  - Max tokens: {adapter.max_context_tokens}")

    # 2. Create prompt
    prompt = TextPrompt(
        content="你是一个写作助手。",
        name="writing-assistant",
        version="v1.0",
        target="更自然",
    )
    print(f"\n✓ Created prompt: {prompt.name} v{prompt.version}")
    print(f"  - Tokens: {adapter.count_tokens(prompt)}")

    # 3. Validate prompt
    issues = adapter.validate_prompt(prompt)
    print(f"\n✓ Validation: {len(issues)} issues")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # 4. Generate response
    response = adapter.generate(prompt)
    print(f"\n✓ Generated response:")
    print(f"  {response}")

    # 5. Create experience with feedback
    experience = ConversationExperience(
        messages=[{"role": "user", "content": "写一首诗"}],
        response="春眠不觉晓...",
        feedback=CompositeFeedback(score=0.3, critique="太老套"),
    )
    print(f"\n✓ Created experience (failure={experience.is_failure})")

    # 6. Compute gradient
    failures = [experience]
    gradient = adapter.compute_gradient(prompt, failures, target="更原创")
    print(f"\n✓ Computed gradient:")
    print(f"  {str(gradient)[:100]}...")

    # 7. Apply gradient
    new_prompt = adapter.apply_gradient(prompt, gradient)
    print(f"\n✓ Applied gradient:")
    print(f"  Version: {prompt.version} → {new_prompt.version}")
    print(f"  New content:\n{new_prompt.content}")

    print("\n" + "=" * 60)
    print(f"Total API calls: {adapter.call_count}")
    print("=" * 60)
