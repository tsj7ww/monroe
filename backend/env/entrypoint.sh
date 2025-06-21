#!/bin/bash
set -e

if [ "$1" = "backend" ]; then
    cd /
    uvicorn app.main:app --host 0.0.0.0 --port $API_PORT --reload --reload-dir app
elif [ "$1" = "jupyter" ]; then
    cd /workspace
    jupyter lab --ip 0.0.0.0 --port $JUPYTER_PORT --no-browser --allow-root \
    --ServerApp.token=$JUPYTER_TOKEN --ServerApp.password=$JUPYTER_PASSWORD
elif [ "$1" = "dask-worker" ]; then
    dask worker $DASK_SCHEDULER_ADDRESS
elif [ "$1" = "ray-worker" ]; then
    ray start --address=$RAY_HEAD_ADDRESS
    tail -f /dev/null # Keep container running
else # Run whatever command was passed
    exec "$@"
fi