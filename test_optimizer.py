"""
优化引擎完整示例

演示如何使用 TrainFreeOptimizer 进行提示词优化。
"""

# 直接从核心模块导入，避免触发适配器依赖
import sys
sys.path.insert(0, '.')

from evoskill.core.prompts import TextPrompt
from evoskill.core.experience import ConversationExperience, CompositeFeedback, FeedbackType
from evoskill.core.gradient import SimpleGradient
from evoskill.core.base_adapter import BaseModelAdapter
from evoskill.core.optimizer import TrainFreeOptimizer
from evoskill.core.optimizer_config import OptimizerConfig
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def create_mock_failures():
    """创建模拟的失败经验"""
    failures = []

    # 失败案例 1: 回答过于啰嗦
    exp1 = ConversationExperience(
        messages=[
            {"role": "user", "content": "什么是Python？"},
        ],
        response="Python是一种高级编程语言，由Guido van Rossum于1991年创建...（后面还有500字）",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CRITIQUE,
            critique="回答过于啰嗦，没有直接回答问题",
            score=0.3,
        ),
    )
    failures.append(exp1)

    # 失败案例 2: 没有给出答案
    exp2 = ConversationExperience(
        messages=[
            {"role": "user", "content": "如何排序列表？"},
        ],
        response="排序是一个复杂的话题，涉及计算机科学的基础...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="没有给出直接答案",
            correction="你可以使用 sorted() 函数或 list.sort() 方法。",
        ),
    )
    failures.append(exp2)

    # 失败案例 3: 格式混乱
    exp3 = ConversationExperience(
        messages=[
            {"role": "user", "content": "解释一下机器学习"},
        ],
        response="机器学习是AI的一个分支它使用算法从数据中学习然后做出预测...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CRITIQUE,
            critique="格式混乱，难以阅读",
            score=0.4,
        ),
    )
    failures.append(exp3)

    return failures


def create_mock_adapter():
    """创建模拟适配器（用于演示，不需要真实API）"""
    class MockAdapter(BaseModelAdapter):
        def __init__(self):
            super().__init__(model_name="mock-model")

        def generate(self, prompt, context=None, **kwargs):
            return "这是一个模拟的回答。"

        def _call_api(self, messages, system=None, temperature=0.7, **kwargs):
            return "Mock API response"

        def _count_tokens_impl(self, text):
            return len(text.split())

        def compute_gradient(self, prompt, failures, target=None, **kwargs):
            """模拟梯度计算"""
            # 分析失败案例，生成改进建议
            critiques = [f.feedback.critique for f in failures if f.feedback.critique]

            gradient_text = "分析失败案例发现：\n"
            gradient_text += "1) 回答过于冗长和啰嗦\n"
            gradient_text += "2) 缺少结构化格式\n"
            gradient_text += "3) 未直接回答用户问题\n"
            gradient_text += "\n建议：简化回答，使用清晰结构，直接给出答案。"

            return SimpleGradient(
                text=gradient_text,
                metadata={"num_failures": len(failures)}
            )

        def apply_gradient(self, prompt, gradient, conservative=False, **kwargs):
            """模拟梯度应用"""
            # 创建新版本的提示词
            new_prompt = prompt.bump_version()

            if conservative:
                # 保守更新：只添加小提示
                new_prompt.content = f"{prompt.content}\n\n注意：回答要简洁直接，格式清晰。"
            else:
                # 激进更新：完全重写
                new_prompt.content = (
                    "你是一个简洁明了的助手。\n\n"
                    "回答原则：\n"
                    "1. 直接回答用户问题，不啰嗦\n"
                    "2. 使用清晰的结构和格式\n"
                    "3. 给出实用的示例和代码\n"
                    "4. 避免过度解释\n\n"
                    "保持回答简洁有力。"
                )

            return new_prompt

    return MockAdapter()


def example_basic_optimization():
    """示例 1: 基本优化流程"""
    print("\n" + "="*80)
    print("示例 1: 基本优化流程")
    print("="*80 + "\n")

    # 1. 创建初始提示词
    initial_prompt = TextPrompt(
        content="你是一个有用的AI助手。",
        version="v1.0",
    )
    print(f"初始提示词 (v{initial_prompt.version}):")
    print(f"{initial_prompt.content}\n")

    # 2. 创建失败经验
    failures = create_mock_failures()
    print(f"收集到 {len(failures)} 个失败案例:")
    for i, f in enumerate(failures, 1):
        critique = f.feedback.critique if f.feedback else "无批评"
        print(f"  {i}. {critique}")
    print()

    # 3. 创建适配器和优化器
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=False,  # 暂时跳过验证
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # 4. 运行优化
    print("开始优化...\n")
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=None,  # 不使用验证器
    )

    # 5. 显示结果
    print("\n" + "-"*80)
    print("优化结果:")
    print("-"*80)
    print(f"执行步数: {result.steps_taken}")
    print(f"是否收敛: {result.converged}")
    print(f"\n最终提示词 (v{result.optimized_prompt.version}):")
    print(f"{result.optimized_prompt.content}\n")

    # 显示历史
    print("优化历史:")
    for step in result.history:
        print(f"  步骤 {step.step_num}: v{step.old_prompt.version} → v{step.new_prompt.version}")
        print(f"    失败数: {step.num_failures}")
        print(f"    梯度摘要: {step.gradient[:60]}...")
        print()


def example_with_validation():
    """示例 2: 带验证的优化"""
    print("\n" + "="*80)
    print("示例 2: 带验证的优化")
    print("="*80 + "\n")

    # 创建初始提示词
    initial_prompt = TextPrompt(
        content="回答用户问题。",
        version="v1.0",
    )
    print(f"初始提示词: {initial_prompt.content}\n")

    # 创建失败经验
    failures = create_mock_failures()

    # 创建验证器函数（模拟）
    def mock_validator(prompt):
        """模拟验证器 - 检查提示词质量"""
        # 假设更好的提示词会：
        # 1. 包含更多指导原则
        # 2. 有清晰的结构

        score = 0.0

        # 检查长度（更详细的提示词得分更高）
        score += min(len(prompt.content) / 300, 0.4)

        # 检查是否包含关键指导词
        keywords = ['直接', '简洁', '结构', '清晰', '原则']
        for kw in keywords:
            if kw in prompt.content:
                score += 0.12

        score = min(score, 1.0)  # 限制在1.0以内

        print(f"  ✓ 验证得分: {score:.3f} (长度={len(prompt.content)}, 关键词={sum(1 for kw in keywords if kw in prompt.content)})")
        return score

    # 创建优化器
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=True,
        early_stopping_patience=2,
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # 运行优化
    print("开始优化（带验证）...\n")
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=mock_validator,
    )

    # 显示结果
    print("\n" + "-"*80)
    print("优化结果:")
    print("-"*80)
    print(f"初始得分: {result.final_score - result.improvement:.3f}")
    print(f"最终得分: {result.final_score:.3f}")
    print(f"总提升: {result.improvement:+.3f}")
    print(f"\n最终提示词:\n{result.optimized_prompt.content}\n")


def example_strategies():
    """示例 3: 不同策略对比"""
    print("\n" + "="*80)
    print("示例 3: 保守 vs 激进策略")
    print("="*80 + "\n")

    initial_prompt = TextPrompt(
        content="帮助用户。",
        version="v1.0",
    )
    failures = create_mock_failures()

    # 保守策略
    print("【保守策略】")
    print("-"*80)
    adapter_conservative = create_mock_adapter()
    config_conservative = OptimizerConfig(
        max_steps=2,
        conservative=True,  # 保守模式
    )
    optimizer_c = TrainFreeOptimizer(adapter_conservative, config_conservative)
    result_c = optimizer_c.optimize(initial_prompt, failures, validator=None)

    print(f"\n最终提示词 (保守):")
    print(f"{result_c.optimized_prompt.content}\n")

    # 激进策略
    print("\n【激进策略】")
    print("-"*80)
    adapter_aggressive = create_mock_adapter()
    config_aggressive = OptimizerConfig(
        max_steps=2,
        conservative=False,  # 激进模式
    )
    optimizer_a = TrainFreeOptimizer(adapter_aggressive, config_aggressive)
    result_a = optimizer_a.optimize(initial_prompt, failures, validator=None)

    print(f"\n最终提示词 (激进):")
    print(f"{result_a.optimized_prompt.content}\n")

    # 对比
    print("\n【对比分析】")
    print("-"*80)
    print(f"保守策略: 提示词长度 {len(result_c.optimized_prompt.content)} 字符")
    print(f"激进策略: 提示词长度 {len(result_a.optimized_prompt.content)} 字符")
    print("\n结论:")
    print("- 保守策略: 小幅修改，保留原有结构")
    print("- 激进策略: 大幅重写，引入新结构")


def main():
    """运行所有示例"""
    print("\n" + "🔍 " * 20)
    print("TrainFreeOptimizer 完整示例")
    print("🔍 " * 20)

    # 运行示例
    example_basic_optimization()
    print("\n" + "="*80 + "\n")

    example_with_validation()
    print("\n" + "="*80 + "\n")

    example_strategies()

    # 总结
    print("\n" + "="*80)
    print("✅ 示例运行完成！")
    print("="*80)
    print("\n核心要点:")
    print("1. TrainFreeOptimizer 使用失败案例进行提示词优化")
    print("2. 支持验证器来评估优化效果")
    print("3. 保守/激进策略控制更新幅度")
    print("4. 早停机制防止过度优化")
    print("\n下一步:")
    print("- 使用真实API适配器（OpenAI/Anthropic）")
    print("- 收集真实的失败案例")
    print("- 设计合适的验证指标")
    print("- 在生产环境中部署优化")


if __name__ == "__main__":
    main()
