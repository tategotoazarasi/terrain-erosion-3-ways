#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100

module purge
module load gcc/14.2.0
module load nvhpc/25.3-gcc14.2.0
#module load openmpi/5.0.8-cuda12.8.0-gcc14.2.0

VENV_DIR="/users/sgzwa126/data/venv"
source $VENV_DIR/bin/activate

NVCC_PATH=$(which nvcc)
if [ -z "$NVCC_PATH" ]; then
    echo "Error: nvcc not found."
    exit 1
fi

NVHPC_ROOT=$(dirname $(dirname $(dirname $NVCC_PATH)))
echo "NVHPC Root: $NVHPC_ROOT"

add_library_path() {
    local lib_pattern=$1
    echo "Looking for $lib_pattern..."
    local paths=$(find $NVHPC_ROOT -name "$lib_pattern" 2>/dev/null | xargs -r -n 1 dirname | sort -u)

    if [ -n "$paths" ]; then
        for p in $paths; do
            if [[ ":$LD_LIBRARY_PATH:" != *":$p:"* ]]; then
                echo "  -> Found in $p, adding to path."
                export LD_LIBRARY_PATH=$p:$LD_LIBRARY_PATH
            fi
        done
    else
        echo "  -> WARNING: Could not find $lib_pattern"
    fi
}

add_library_path "libnvrtc.so*"
add_library_path "libcurand.so*"
add_library_path "libcufft.so*"

# 6. Run Benchmark
python step3_generate_and_infer.py