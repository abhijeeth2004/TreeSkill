"""全自动化优化示例 - 无需人工干预

这个示例演示如何实现完全自动化的提示词优化循环。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
    TrainFreeOptimizer,
    OptimizerConfig,
    AutoValidator,
)
from evoskill.adapters import OpenAIAdapter
from evoskill.tools import tool, tool_registry


# ===========================================================================
# Step 1: 定义自动评估工具
# ===========================================================================

@tool(
    name="evaluate_response",
    schema={
        "type": "object",
        "properties": {
            "user_input": {"type": "string", "description": "用户输入"},
            "agent_response": {"type": "string", "description": "助手回复"},
            "criteria": {"type": "string", "description": "评估标准"}
        },
        "required": ["user_input", "agent_response", "criteria"]
    }
)
def evaluate_response(user_input: str, agent_response: str, criteria: str) -> dict:
    """自动评估助手回复质量

    使用LLM作为Judge来评估回复是否符合标准
    """
    # 这里可以使用另一个LLM作为Judge
    # 简化示例：基于规则的评估
    score = 0.5
    critique = None

    # 检查长度
    if len(agent_response) > 500:
        score -= 0.2
        critique = "回复过长"

    # 检查是否包含示例
    if "例如" in agent_response or "示例" in agent_response:
        score += 0.1

    # 检查是否回答了问题
    if "?" in user_input and ("是" in agent_response or "不是" in agent_response):
        score += 0.2

    return {
        "score": max(0, min(1, score)),
        "critique": critique,
        "is_positive": score >= 0.5
    }


# ===========================================================================
# Step 2: 定义测试用例生成器
# ===========================================================================

def generate_test_cases() -> list:
    """自动生成测试用例"""
    return [
        {
            "user_input": "什么是Python？",
            "expected_criteria": "简洁、有结构、有示例"
        },
        {
            "user_input": "如何排序列表？",
            "expected_criteria": "直接回答、提供代码示例"
        },
        {
            "user_input": "解释机器学习",
            "expected_criteria": "通俗易懂、有类比、不要太长"
        }
    ]


# ===========================================================================
# Step 3: 自动收集失败案例
# ===========================================================================

def collect_failures_automatically(
    adapter,
    prompt,
    test_cases,
    threshold=0.5
) -> list:
    """自动运行测试用例并收集失败案例"""
    failures = []

    for i, test in enumerate(test_cases):
        print(f"\n测试用例 {i+1}/{len(test_cases)}: {test['user_input'][:30]}...")

        # 生成回复
        exp = ConversationExperience(
            messages=[{"role": "user", "content": test["user_input"]}],
            response=adapter.generate(prompt).content if hasattr(adapter.generate(prompt), 'content') else str(adapter.generate(prompt))
        )

        # 自动评估
        eval_result = tool_registry.execute(
            "evaluate_response",
            user_input=test["user_input"],
            agent_response=exp.response,
            criteria=test["expected_criteria"]
        )

        # 如果不合格，添加到失败案例
        if eval_result["score"] < threshold:
            feedback = CompositeFeedback(
                feedback_type=FeedbackType.CRITIQUE,
                critique=eval_result["critique"],
                score=eval_result["score"]
            )
            exp.feedback = feedback
            failures.append(exp)
            print(f"  ❌ 失败 (得分: {eval_result['score']:.2f}): {eval_result['critique']}")
        else:
            print(f"  ✓ 通过 (得分: {eval_result['score']:.2f})")

    return failures


# ===========================================================================
# Step 4: 完全自动化的优化循环
# ===========================================================================

def auto_optimize_loop(
    adapter,
    initial_prompt,
    max_iterations=3,
    convergence_threshold=0.8
):
    """完全自动化的优化循环 - 无需人工干预"""
    print("="*80)
    print("🤖 开始全自动化优化循环")
    print("="*80)

    current_prompt = initial_prompt
    test_cases = generate_test_cases()

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}/{max_iterations}")
        print(f"{'='*80}")

        # 1. 自动收集失败案例
        failures = collect_failures_automatically(adapter, current_prompt, test_cases)

        if not failures:
            print(f"\n✅ 所有测试通过！优化完成。")
            break

        print(f"\n发现 {len(failures)} 个失败案例，开始优化...")

        # 2. 运行优化
        config = OptimizerConfig(
            max_steps=2,
            conservative=False,
            validate_every_step=False  # 已经在上面验证过了
        )
        optimizer = TrainFreeOptimizer(adapter, config)

        result = optimizer.optimize(current_prompt, failures)

        # 3. 更新提示词
        current_prompt = result.optimized_prompt

        print(f"\n优化完成:")
        print(f"  版本: {initial_prompt.version} → {current_prompt.version}")
        print(f"  步数: {result.steps_taken}")

        # 4. 可选：再次验证
        print(f"\n验证优化后的提示词...")
        new_failures = collect_failures_automatically(adapter, current_prompt, test_cases)

        success_rate = 1 - (len(new_failures) / len(test_cases))
        print(f"\n成功率: {success_rate:.1%}")

        if success_rate >= convergence_threshold:
            print(f"\n✅ 达到收敛阈值 ({convergence_threshold:.1%})，优化完成！")
            break

    print(f"\n{'='*80}")
    print("🎉 全自动化优化完成")
    print(f"{'='*80}")
    print(f"最终版本: {current_prompt.version}")
    print(f"最终提示词:\n{current_prompt.content[:200]}...")

    return current_prompt


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """运行全自动化优化示例"""

    # 初始化
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    initial_prompt = TextPrompt(
        content="你是一个有用的助手。",
        version="v1.0"
    )

    # 运行完全自动化的优化
    optimized_prompt = auto_optimize_loop(
        adapter=adapter,
        initial_prompt=initial_prompt,
        max_iterations=3,
        convergence_threshold=0.8
    )

    # 保存结果
    print(f"\n最终优化后的提示词:")
    print("="*80)
    print(optimized_prompt.content)
    print("="*80)


if __name__ == "__main__":
    main()
