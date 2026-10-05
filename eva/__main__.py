"""Allow running EVA as a module: python -m eva"""

import asyncio

from eva.cli import run_cli


def main() -> None:
    """Entry point for the eva console script and python -m eva."""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
