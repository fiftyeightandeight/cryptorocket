#!/usr/bin/env bash
# Download the latest crypto.duckdb from the GitHub Release.
# Usage: ./scripts/pull_db.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCRIPT_DIR}/../data/crypto.duckdb"

echo "Downloading latest database..."
gh release download data-latest -p crypto.duckdb -D "$(dirname "$DEST")" --clobber
echo "Saved to ${DEST} ($(du -h "$DEST" | cut -f1))"
