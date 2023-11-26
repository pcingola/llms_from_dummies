#!/bin/bash -eu
set -o pipefail

DIR="$( cd "$( dirname "$0" )" && pwd )"
source "$DIR/config.sh"

cd "$SRC_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$LM_HOME/lit-llama"

mkdir -p "$LOGS_DIR"

echo "SERVER_PORT: ${SERVER_PORT}"

uvicorn app.main:app \
    --reload \
    --host "0.0.0.0" \
    --port $SERVER_PORT \
    2>&1 | tee "$LOGS_DIR/server.log"
