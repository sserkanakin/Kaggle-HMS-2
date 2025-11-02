#!/bin/bash
#
# Optimized launcher for graph preprocessing on 20-core system
# This script ensures best performance with stability
#

set -e  # Exit on error

echo "=============================================="
echo "HMS Graph Preprocessing - Optimized Launcher"
echo "=============================================="
echo ""
echo "System: $(nproc) CPU cores detected"
echo "RAM: $(free -h | grep Mem | awk '{print $2}') total"
echo ""

# Change to project directory
cd /root/Kaggle-HMS-2

# Activate conda environment if needed
if [ -d "/root/.local/share/mamba/envs/graph-ml-2" ]; then
    echo "Activating conda environment: graph-ml-2"
    source /root/.local/share/mamba/bin/activate graph-ml-2 2>/dev/null || true
fi

# Avoid over-threading inside each worker so multiple processes scale well
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# Run with optimal settings for 20-core system
# Using 8 workers for best balance of speed and stability
echo ""
echo "Starting preprocessing with 8 workers..."
echo "Estimated time: 3-4 hours for full dataset"
echo ""

# Prefer environment python if available
if command -v python &>/dev/null; then
    python src/data/make_graph_dataset.py --workers 16
else
    python3 src/data/make_graph_dataset.py --workers 16
fi

echo ""
echo "=============================================="
echo "Preprocessing complete!"
echo "=============================================="
