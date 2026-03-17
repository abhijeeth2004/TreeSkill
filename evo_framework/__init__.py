"""
Evo-Framework (Legacy)

⚠️ DEPRECATED: This module is deprecated.
Please use 'evoskill' instead.

This module is kept for backward compatibility only.
All imports are redirected to 'evoskill'.

Migration Guide:
---------------
Old:
    from evo_framework import Skill, Trace

New:
    from evoskill import Skill, Trace  # Same API

The migration is transparent - all your existing code will continue to work.
"""

# Re-export everything from evoskill for backward compatibility
import warnings

warnings.warn(
    "The 'evo_framework' package is deprecated. "
    "Please use 'evoskill' instead. "
    "Your code will continue to work, but please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all from new package
try:
    from evoskill import *
    from evoskill import (
        # Version info
        __version__,
        __author__,
        __email__,
    )
except ImportError as e:
    raise ImportError(
        "Cannot import from 'evoskill'. Please ensure evoskill is installed: "
        "pip install evoskill"
    ) from e

# Legacy imports (for old code)
__all__ = [
    # Core abstraction layer (new)
    "OptimizablePrompt",
    "TextualGradient",
    "Experience",
    "Feedback",
    "ModelAdapter",
    "TextPrompt",
    "MultimodalPrompt",
    "StructuredPrompt",
    "SimpleGradient",
    "MultimodalGradient",
    "GradientHistory",
    "ConversationExperience",
    "MultimodalExperience",
    "CompositeFeedback",
    "FeedbackType",
    "BaseModelAdapter",

    # Model adapters (new)
    "OpenAIAdapter",
    "AnthropicAdapter",

    # Registry (new)
    "registry",
    "adapter",
    "optimizer",
    "hook",
    "ComponentMeta",

    # Legacy (v0.1 - backward compatibility)
    "CheckpointManager",
    "ContentPart",
    "LegacyFeedback",
    "GlobalConfig",
    "ImageContent",
    "Message",
    "Registry",
    "Skill",
    "SkillMeta",
    "SkillTree",
    "TextContent",
    "Trace",
    "registry",

    # Version
    "__version__",
    "__author__",
    "__email__",
]

# Maintain version compatibility
__version__ = "0.2.0"  # Keep in sync with evoskill
