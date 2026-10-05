"""EVA CLI — interactive command-line interface.

Provides a minimal interactive loop for Milestone 1A.
Uses the mock provider for responses when no real provider
is configured.
"""

from __future__ import annotations

import asyncio
import sys

from eva.core.enums import EVAStatus
from eva.logging import get_logger
from eva.runtime import EVARuntime

logger = get_logger(__name__)

BANNER = """\
====================================
           E . V . A
   Evolving Virtual Autonomy
===================================="""


def _print_status(runtime: EVARuntime) -> None:
    """Display EVA status summary."""
    settings = runtime.settings
    state = runtime.state
    print()
    print(BANNER)
    print()
    print(f"  Status:       {state.status.value.upper()}")
    print(f"  Environment:  {settings.env.value}")
    print(f"  Provider:     {settings.model_provider}")
    print(f"  Mode:         {state.current_mode.value}")
    print(f"  Safety:       ACTIVE")
    print(f"  Memory:       NOT CONNECTED")
    print(f"  Tools:        NONE")
    print()
    print("  EVA is ready.")
    print()


async def _interaction_loop(runtime: EVARuntime) -> None:
    """Run the main interactive loop."""
    _print_status(runtime)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("You: ")
            )
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = user_input.strip()

        if not stripped:
            continue

        if stripped.lower() in ("exit", "quit"):
            break

        # Track conversation turn
        runtime.context.increment_turn()

        # Generate response via provider
        response = await runtime.provider.generate(stripped)
        print(f"\nEVA: {response}\n")


async def run_cli() -> None:
    """Entry point for the EVA CLI."""
    runtime = EVARuntime()

    try:
        await runtime.start()
        await _interaction_loop(runtime)
    except Exception:
        logger.exception("Fatal error in EVA CLI")
        sys.exit(1)
    finally:
        await runtime.stop()
