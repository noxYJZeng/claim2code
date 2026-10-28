#!/usr/bin/env bash
# ============================================================
# Claim2Code — start the local stack
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# 1. ensure .env exists
if [ ! -f "$ROOT/.env" ]; then
    echo "› .env not found — copying from .env.example"
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "› EDIT $ROOT/.env for database and local or hosted model access before live analysis"
fi

# 2. ensure data directories exist (for cloned repos / uploads / embeddings)
mkdir -p "$ROOT/data/repos" "$ROOT/data/uploads" "$ROOT/data/embeddings"
touch "$ROOT/data/repos/.gitkeep" "$ROOT/data/uploads/.gitkeep" "$ROOT/data/embeddings/.gitkeep"

# 3. bring up the stack
docker compose up -d --build

# Run migrations from their actual working directory before declaring
# the stack ready; SELECT 1 alone does not prove application tables exist.
docker compose exec -T --workdir /app/api api alembic upgrade head

echo ""
echo "Claim2Code stack is up:"
echo "  Web:           http://localhost:${WEB_PORT:-5173}"
echo "  API:           http://localhost:${API_PORT:-8000}"
echo "  API docs:      http://localhost:${API_PORT:-8000}/docs"
echo "  Postgres:      localhost:${POSTGRES_PORT:-5432}"
echo ""
echo "Follow logs with: docker compose logs -f"
