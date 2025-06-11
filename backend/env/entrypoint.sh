#!/bin/bash
set -e

# Check if we need to start Ray or Dask services
if [ "$1" = "ray-head" ]; then
    # Start Ray head node
    ray start --head --port=$RAY_HEAD_PORT --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
    # Keep container running
    tail -f /dev/null
elif [ "$1" = "ray-worker" ]; then
    # Start Ray worker node, needs RAY_HEAD_ADDRESS env var
    ray start --address=$RAY_HEAD_ADDRESS
    # Keep container running
    tail -f /dev/null
elif [ "$1" = "dask-scheduler" ]; then
    # Start Dask scheduler
    dask scheduler
elif [ "$1" = "dask-worker" ]; then
    # Start Dask worker, needs DASK_SCHEDULER_ADDRESS env var
    dask worker $DASK_SCHEDULER_ADDRESS
elif [ "$1" = "jupyter" ]; then
    # Start Jupyter for development
    jupyter lab --ip 0.0.0.0 --port $JUPYTER_PORT --no-browser --allow-root \
    --ServerApp.token=$JUPYTER_TOKEN --ServerApp.password=$JUPYTER_PASSWORD
else
    # Run whatever command was passed
    exec "$@"
fi