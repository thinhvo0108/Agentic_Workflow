#!/usr/bin/env bash
# Configure git to use the committed hooks in .githooks/.
# Run once after cloning: bash scripts/install-hooks.sh
set -euo pipefail

git config core.hooksPath .githooks
echo "Git hooks installed from .githooks/ — pre-commit checks are now active."
