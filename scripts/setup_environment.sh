#!/usr/bin/env bash
# Convenience wrapper that runs the project setup steps.
# Equivalent to: uv sync && uv run polyfun-gpn setup
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
uv run polyfun-gpn setup "$@"
