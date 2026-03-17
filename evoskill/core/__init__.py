"""Core abstraction layer for train-free prompt optimization.

This module provides the foundational abstractions that make the framework
model-agnostic and multimodal-ready.
"""

from evoskill.core.abc import (
    OptimizablePrompt,
    TextualGradient,
    Experience,
    Feedback,
    ModelAdapter,
    PromptSerializer,
)
from evoskill.core.prompts import (
    TextPrompt,
    MultimodalPrompt,
    StructuredPrompt,
)
from evoskill.core.gradient import (
    SimpleGradient,
    MultimodalGradient,
    GradientHistory,
)
from evoskill.core.experience import (
    ConversationExperience,
    MultimodalExperience,
    CompositeFeedback,
    FeedbackType,
)
from evoskill.core.base_adapter import BaseModelAdapter
from evoskill.core.optimizer import TrainFreeOptimizer
from evoskill.core.optimizer_config import (
    OptimizerConfig,
    OptimizationResult,
    OptimizationStep,
    Validator,
)
from evoskill.core.strategies import (
    OptimizationStrategy,
    ConservativeStrategy,
    AggressiveStrategy,
    AdaptiveStrategy,
    get_strategy,
)
from evoskill.core.validators import (
    AutoValidator,
    MetricValidator,
    CompositeValidator,
    create_simple_validator,
    create_metric_validator,
)

__all__ = [
    # Abstract base classes
    "OptimizablePrompt",
    "TextualGradient",
    "Experience",
    "Feedback",
    "ModelAdapter",
    "PromptSerializer",
    # Base adapter
    "BaseModelAdapter",
    # Concrete prompt types
    "TextPrompt",
    "MultimodalPrompt",
    "StructuredPrompt",
    # Gradient types
    "SimpleGradient",
    "MultimodalGradient",
    "GradientHistory",
    # Experience types
    "ConversationExperience",
    "MultimodalExperience",
    "CompositeFeedback",
    "FeedbackType",
    # Optimizer
    "TrainFreeOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "OptimizationStep",
    "Validator",
    # Strategies
    "OptimizationStrategy",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "AdaptiveStrategy",
    "get_strategy",
    # Validators
    "AutoValidator",
    "MetricValidator",
    "CompositeValidator",
    "create_simple_validator",
    "create_metric_validator",
]
