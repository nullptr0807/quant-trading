"""Project-local scripts package.

Keeping this directory an explicit package prevents legacy modules that prepend
``~/quant-trading`` from merging two namespace-package directories and silently
loading a production script while testing an isolated worktree.
"""
