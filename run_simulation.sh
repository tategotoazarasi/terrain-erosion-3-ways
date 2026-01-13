#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-v100

# 1. Load Modules
module purge
module load gcc/14.2.0
module load nvhpc/25.3-gcc14.2.0

# 2. Activate Environment
VENV_DIR="/users/sgzwa126/data/venv"
source $VENV_DIR/bin/activate

# 3. Fix Environment for CuPy on NVHPC (Comprehensive Fix)
# 获取 nvcc 的路径
NVCC_PATH=$(which nvcc)
if [ -z "$NVCC_PATH" ]; then
    echo "Error: nvcc not found."
    exit 1
fi

# 获取 NVHPC 安装根目录 (例如 .../Linux_x86_64/25.3)
NVHPC_ROOT=$(dirname $(dirname $(dirname $NVCC_PATH)))
echo "NVHPC Root: $NVHPC_ROOT"

# 定义一个函数，在 NVHPC 根目录下查找特定的库文件，并将包含该库的目录加入 LD_LIBRARY_PATH
add_library_path() {
    local lib_pattern=$1
    echo "Looking for $lib_pattern..."
    # 使用 find 查找文件，获取目录名，去重
    local paths=$(find $NVHPC_ROOT -name "$lib_pattern" 2>/dev/null | xargs -r -n 1 dirname | sort -u)

    if [ -n "$paths" ]; then
        for p in $paths; do
            # 只有当路径不在 LD_LIBRARY_PATH 中时才添加
            if [[ ":$LD_LIBRARY_PATH:" != *":$p:"* ]]; then
                echo "  -> Found in $p, adding to path."
                export LD_LIBRARY_PATH=$p:$LD_LIBRARY_PATH
            fi
        done
    else
        echo "  -> WARNING: Could not find $lib_pattern"
    fi
}

# 查找 CuPy 运行所需的关键 CUDA 库
# libnvrtc: 运行时编译 (Runtime Compilation)
add_library_path "libnvrtc.so*"
# libcurand: 随机数生成 (Random Number Generation) - 你的报错就是缺这个
add_library_path "libcurand.so*"
# libcufft: 快速傅里叶变换 (FFT) - util.py 中使用了 FFT
add_library_path "libcufft.so*"

# 4. Debug Info
echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 5. Optimize Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 6. Run Benchmark
python simulation.py
python make_hillshaded_image.py simulation.npy output.png
echo "DONE"