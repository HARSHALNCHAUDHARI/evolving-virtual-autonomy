"""EVA brain — cognitive loop and central processing.

The brain package contains the cognitive engine and all pipeline
components: understanding, reasoning, planning, decision, execution,
verification, and response generation.
"""

from eva.core.brain.engine import CognitiveEngine
from eva.core.brain.models import CognitiveResult

__all__ = ["CognitiveEngine", "CognitiveResult"]
