#!/usr/bin/env python3
"""
加载已存储的 Skill 并使用 Config 的完整示例

这个示例演示如何：
1. 加载已存储的 skill 文件
2. 使用全局配置
3. 使用适配器调用 LLM
4. 保存优化后的 skill

流程：
- 加载配置 → 加载 skill → 创建适配器 → 调用 LLM → 保存 skill
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill import (
    # 配置
    GlobalConfig,
    # Skill 管理
    load_skill,
    save_skill,
    compile_messages,
    # 新架构（可选）
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_load_config_and_skill():
    """示例 1: 加载配置和 Skill"""
    print("\n" + "="*60)
    print("示例 1: 加载配置和 Skill")
    print("="*60 + "\n")

    # 方式 1: 从 YAML 文件加载配置
    config_path = "demo/example/config.yaml"
    if Path(config_path).exists():
        config = GlobalConfig.from_yaml(config_path)
        print(f"✓ 从 {config_path} 加载配置")
        print(f"  - LLM 模型: {config.llm.model}")
        print(f"  - Judge 模型: {config.llm.judge_model}")
        print(f"  - API 地址: {config.llm.base_url}")
        print(f"  - Trace 路径: {config.storage.trace_path}")
    else:
        print(f"⚠ 配置文件 {config_path} 不存在，使用默认配置")
        config = GlobalConfig()

    # 方式 2: 从 .env 文件加载（自动）
    # 只需要在项目根目录创建 .env 文件，GlobalConfig 会自动读取
    # .env 示例：
    #   EVO_LLM_API_KEY=your-key
    #   EVO_LLM_BASE_URL=https://api.siliconflow.cn/v1
    #   EVO_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct

    # 方式 3: 从环境变量加载（最高优先级）
    # export EVO_LLM_API_KEY="your-key"
    # config = GlobalConfig()  # 会自动读取环境变量

    print("\n" + "-"*60)

    # 加载 Skill
    skill_path = "demo/example/skill.yaml"
    if Path(skill_path).exists():
        skill = load_skill(skill_path)
        print(f"✓ 从 {skill_path} 加载 Skill")
        print(f"  - 名称: {skill.name}")
        print(f"  - 版本: {skill.version}")
        print(f"  - System Prompt 长度: {len(skill.system_prompt)} 字符")
        print(f"  - Target: {skill.target}")
    else:
        print(f"⚠ Skill 文件 {skill_path} 不存在")
        return None

    return config, skill


def example_2_use_adapter(config, skill):
    """示例 2: 使用适配器调用 LLM"""
    print("\n" + "="*60)
    print("示例 2: 使用适配器调用 LLM")
    print("="*60 + "\n")

    # 延迟导入 OpenAIAdapter（需要安装 openai tiktoken）
    try:
        from evoskill import OpenAIAdapter
    except ImportError:
        print("⚠ OpenAIAdapter 导入失败")
        print("  需要安装依赖:")
        print("    pip install openai tiktoken")
        print("  或:")
        print("    pip install -e .\n")
        print("  跳过此示例\n")
        return None

    # 创建 OpenAI 适配器
    adapter = OpenAIAdapter(
        api_key=config.llm.api_key.get_secret_value(),
        base_url=config.llm.base_url,
        model=config.llm.model,
    )

    print(f"✓ 创建 OpenAI 适配器")
    print(f"  - API 地址: {config.llm.base_url}")
    print(f"  - 模型: {config.llm.model}")

    # 准备用户输入
    user_input = [{"role": "user", "content": "帮我写一段关于春天的短文"}]

    # 编译消息（system prompt + few-shot + user input）
    messages = compile_messages(skill, user_input)

    print(f"\n✓ 编译消息")
    print(f"  - 消息数量: {len(messages)}")
    print(f"  - 第一条消息角色: {messages[0]['role']}")

    # 调用 LLM
    print(f"\n→ 调用 LLM...")
    try:
        response = adapter.generate(messages, temperature=config.llm.temperature)
        print(f"✓ 收到响应")
        print(f"\n助手回复:\n{response.content}\n")
        return response
    except Exception as e:
        print(f"✗ 调用失败: {e}")
        print(f"  提示: 请检查 API Key 和 base_url 是否正确")
        return None


def example_3_save_skill(skill):
    """示例 3: 保存 Skill"""
    print("\n" + "="*60)
    print("示例 3: 保存 Skill")
    print("="*60 + "\n")

    # 修改 skill（模拟优化后）
    skill.version = "v1.1"
    skill.system_prompt = skill.system_prompt + "\n\n新增规则：避免使用过于正式的客套话。"

    # 保存到新路径
    output_path = Path("skills/my-skill-v1.1.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_skill(skill, output_path)

    print(f"✓ Skill 已保存到 {output_path}")
    print(f"  - 名称: {skill.name}")
    print(f"  - 版本: {skill.version}")
    print(f"  - 文件大小: {output_path.stat().st_size} 字节")


def example_4_manual_feedback_and_gradient():
    """示例 4: 手动反馈 + 计算梯度（新架构）"""
    print("\n" + "="*60)
    print("示例 4: 手动反馈 + 计算梯度（新架构）")
    print("="*60 + "\n")

    # 创建初始 Prompt
    prompt = TextPrompt(
        content="你是一个写作助手，帮助用户撰写各种文本。"
    )

    print(f"✓ 初始 Prompt:")
    print(f"  {prompt.content}\n")

    # 创建失败经验（模拟 /bad 和 /rewrite 命令）
    experience = ConversationExperience(
        messages=[
            {"role": "user", "content": "写一段关于春天的短文"},
        ],
        response="春天来了，万物复苏...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="回答太简单，缺乏细节和感情",
            correction="春天来了，小区的玉兰花开了，空气中弥漫着淡淡的花香...",
        ),
    )

    print(f"✓ 创建失败经验:")
    print(f"  - 用户输入: {experience.messages[0]['content']}")
    print(f"  - 助手回复: {experience.response}")
    print(f"  - 反馈类型: {experience.feedback.feedback_type}")
    print(f"  - 批评: {experience.feedback.critique}")
    print(f"  - 理想回答: {experience.feedback.correction}\n")

    # 使用 Mock 适配器计算梯度
    try:
        from evoskill import MockAdapter
        adapter = MockAdapter()

        print(f"→ 计算文本梯度...")
        gradient = adapter.compute_gradient(
            prompt=prompt,
            failures=[experience],  # 失败经验列表
        )

        print(f"✓ 梯度计算完成:")
        print(f"  - 梯度内容:\n{gradient.text}\n")

        # 应用梯度更新 Prompt
        print(f"→ 应用梯度更新 Prompt...")
        new_prompt = adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
        )

        print(f"✓ 新 Prompt:")
        print(f"  {new_prompt.content}\n")

        print(f"✓ 优化完成！")
    except ImportError as e:
        print(f"⚠ MockAdapter 导入失败: {e}")
        print(f"  跳过梯度计算示例\n")




def example_5_load_and_continue():
    """示例 5: 加载已有 Skill 并继续优化"""
    print("\n" + "="*60)
    print("示例 5: 加载已有 Skill 并继续优化")
    print("="*60 + "\n")

    skill_path = Path("skills/my-skill-v1.1.yaml")

    if not skill_path.exists():
        print(f"⚠ Skill 文件 {skill_path} 不存在")
        print(f"  请先运行前面的示例创建 skill")
        return

    # 加载已有 skill
    skill = load_skill(skill_path)

    print(f"✓ 加载已有 Skill:")
    print(f"  - 名称: {skill.name}")
    print(f"  - 版本: {skill.version}")
    print(f"  - System Prompt 长度: {len(skill.system_prompt)} 字符\n")

    # 继续优化（模拟）
    skill.version = "v1.2"
    skill.system_prompt = skill.system_prompt + "\n\n新增规则：多用生动的比喻和细节描写。"

    # 保存新版本
    new_path = Path("skills/my-skill-v1.2.yaml")
    save_skill(skill, new_path)

    print(f"✓ 优化后保存到 {new_path}")
    print(f"  - 新版本: {skill.version}\n")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("EvoSkill - 加载已存储 Skill 并使用 Config 示例")
    print("="*60)

    # 示例 1: 加载配置和 Skill
    result = example_1_load_config_and_skill()
    if result is None:
        print("\n⚠ 缺少必要文件，示例终止")
        return

    config, skill = result

    # 示例 2: 使用适配器调用 LLM
    # 注意：需要有效的 API Key 才能成功
    # example_2_use_adapter(config, skill)

    # 示例 3: 保存 Skill
    example_3_save_skill(skill)

    # 示例 4: 手动反馈 + 计算梯度
    example_4_manual_feedback_and_gradient()

    # 示例 5: 加载并继续优化
    example_5_load_and_continue()

    print("\n" + "="*60)
    print("所有示例完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
