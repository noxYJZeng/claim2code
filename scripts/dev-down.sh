#!/usr/bin/env bash
# ============================================================
# Claim2Code — stop the local stack
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose down