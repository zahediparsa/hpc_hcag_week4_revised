#!/usr/bin/env bash
#SBATCH --job-name=ds4se_w4_hcag_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-unknown} | Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "========================================================"

module reset
module load lang/Python/3.11.3-GCCcore-12.3.0

export WORK_ROOT="/scratch/hpc-prf-dssecs/zahedi/ds4se_group3" #Change it to your username
export CODE_DIR="$WORK_ROOT/code"
export INPUT_DIR="$WORK_ROOT/input"
export OUTPUT_ROOT="$WORK_ROOT/week4_hcag_all_algorithms_output"
export HADOOP_DIR="$WORK_ROOT/hadoop"
export VENV_DIR="$WORK_ROOT/venvs/ds4se"

export CLUSTERS_DIR="$INPUT_DIR"
export ARC_CLUSTERS_CSV="$INPUT_DIR/clusters_arc.csv"
export ACDC_CLUSTERS_CSV="$INPUT_DIR/clusters_acdc.csv"
export LIMBO_CLUSTERS_CSV="$INPUT_DIR/clusters_limbo.csv"

export SOURCE_CODE_DIR="$HADOOP_DIR/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce"

export HF_HOME="$WORK_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$WORK_ROOT/torch_cache"
export PIP_CACHE_DIR="$WORK_ROOT/pip_cache"
export TOKENIZERS_PARALLELISM=false

export MODEL_NAME="${MODEL_NAME:-ibm-granite/granite-34b-code-instruct-8k}"
export DO_SAMPLE="${DO_SAMPLE:-false}"
export RESUME="${RESUME:-true}"

mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

source "$VENV_DIR/bin/activate"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[error] HF_TOKEN is not set."
    echo "Submit with: HF_TOKEN=hf_... sbatch week4_hcag_all_algorithms_run.sh"
    exit 1
fi

for f in "$ARC_CLUSTERS_CSV" "$ACDC_CLUSTERS_CSV" "$LIMBO_CLUSTERS_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "[error] Missing cluster CSV: $f"
        exit 1
    fi
done

if [[ ! -d "$SOURCE_CODE_DIR" ]]; then
    echo "[warn] SOURCE_CODE_DIR not found. Attempting to clone Hadoop (trunk) into $HADOOP_DIR ..."
    mkdir -p "$HADOOP_DIR"
    git clone --depth 1 --branch rel/release-3.4.1 https://github.com/apache/hadoop.git "$HADOOP_DIR"
    if [[ ! -d "$SOURCE_CODE_DIR" ]]; then
        echo "[error] Hadoop was cloned but SOURCE_CODE_DIR is still missing: $SOURCE_CODE_DIR"
        echo "Check that HADOOP_DIR and SOURCE_CODE_DIR paths are correct."
        exit 1
    fi
    echo "[info] Hadoop clone complete."
fi

if [[ ! -f "$CODE_DIR/week4_hcag_hpc_all_algorithms.py" ]]; then
    echo "[error] Missing Python script: $CODE_DIR/week4_hcag_hpc_all_algorithms.py"
    exit 1
fi

echo "WORK_ROOT=$WORK_ROOT"
echo "CODE_DIR=$CODE_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "ARC_CLUSTERS_CSV=$ARC_CLUSTERS_CSV"
echo "ACDC_CLUSTERS_CSV=$ACDC_CLUSTERS_CSV"
echo "LIMBO_CLUSTERS_CSV=$LIMBO_CLUSTERS_CSV"
echo "HADOOP_DIR=$HADOOP_DIR"
echo "SOURCE_CODE_DIR=$SOURCE_CODE_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "MODEL_NAME=$MODEL_NAME"

python - <<'PY'
import sys
import torch
import transformers
import pandas
import accelerate

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Pandas:", pandas.__version__)
print("Accelerate:", accelerate.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

nvidia-smi

srun python "$CODE_DIR/week4_hcag_hpc_all_algorithms.py"

echo "========================================================"
echo "Job finished: $(date)"
echo "========================================================"
