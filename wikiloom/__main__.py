"""Module entry point so ``python -m wikiloom`` runs the CLI.

Lets MCP client configs (Claude Desktop, Claude Code, etc.) launch
the server without depending on the ``wikiloom`` script being on PATH.
"""

from wikiloom.cli import main

main()
