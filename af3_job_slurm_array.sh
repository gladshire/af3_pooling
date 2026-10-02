#!/bin/bash
#SBATCH -J af3_pool_run
#SBATCH -o logs/af3_%j.o
#SBATCH -e logs/af3_%j.e
#SBATCH -p h100
#SBATCH -N 1                        # use the full per-job/per-user allowance
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # confirm against actual node core count (see note)
#SBATCH --mem=240G
#SBATCH -t 12:00:00                  # h100's MaxWall = 48:00:00
#SBATCH --mail-user=mwoodc2@uic.edu
#SBATCH --mail-type=all
#SBATCH -A TG-BIO260148


module load tacc-apptainer


BASE_DIR="/work2/11699/mwoodcockgirard/stampede3/af3_workspace"
RUN_AF3_PATH="/app/alphafold/run_alphafold.py"
AF3_DIR="$BASE_DIR/alphafold3"
SIF_PATH="$AF3_DIR/alphafold3_tacc.sif"
INPUT_DIR="$BASE_DIR/af3_input"
OUTPUT_DIR="$BASE_DIR/output_af3"
PARAM_DIR="$BASE_DIR/af3_params"
ORDER_FILE="$BASE_DIR/pool_order.txt"
LOG_DIR="$BASE_DIR/logs"
#CURSOR_FILE="$BASE_DIR/af3_workspace/cursor.txt"
#LOCK_DIR="$BASE_DIR/af3_workspace/cursor.lockdir"
CURSOR_FILE="$BASE_DIR/cursor_small.txt"
LOCK_DIR="$BASE_DIR/cursor_small.lockdir"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
[ -f "$CURSOR_FILE" ] || echo 1 > "$CURSOR_FILE"

NGPUS=4

claim_batch() {
    local n="$1"
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 0.2; done
    local start total end
    start=$(cat "$CURSOR_FILE")
    total=$(wc -l < "$ORDER_FILE")
    if (( start > total )); then rmdir "$LOCK_DIR"; return; fi
    end=$(( start + n - 1 ))
    (( end > total )) && end=$total
    echo $(( end + 1 )) > "$CURSOR_FILE"
    sed -n "${start},${end}p" "$ORDER_FILE"
    rmdir "$LOCK_DIR"
}

run_af3() {
    local POOL_ID="$1" GPU_SLOT="$2"
    local STEM="pool_${POOL_ID}"
    local INPUT_JSON="${STEM}.json"
    local OUT_CIF="$OUTPUT_DIR/$STEM/${STEM}_model.cif"
    local LOG_FILE="$LOG_DIR/${STEM}_log.txt"

    if [ ! -f "$INPUT_DIR/$INPUT_JSON" ]; then
        echo "No JSON for $STEM. Skipping ..." >> "$LOG_FILE"; return
    fi
    if [ -f "$OUT_CIF" ]; then
        echo "Model already exists for $STEM. Skipping ..." >> "$LOG_FILE"; return
    fi

#    CUDA_VISIBLE_DEVICES=$GPU_SLOT apptainer exec --nv \
#	--env XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
#	--env XLA_PYTHON_CLIENT_PREALLOCATE=false \
#	--env TF_FORCE_UNIFIED_MEMORY=true \
#	--env XLA_CLIENT_MEM_FRACTION=3.2 \
#        --bind "$INPUT_DIR":/root/input_af3 \
#        --bind "$OUTPUT_DIR":/root/output_af3 \
#        --bind "$PARAM_DIR":/root/models \
#	--bind /work2/11699/mwoodcockgirard/stampede3/jax_cache/:/root/jax_cache \
#        "$AF3_DIR/alphafold3_tacc.sif" \
#        python3 $RUN_AF3_PATH \
#	--jax_compilation_cache_dir=/root/jax_cache \
#	--buckets 5120 \
#        --norun_data_pipeline \
#        --json_path=/root/input_af3/${INPUT_JSON} \
#        --model_dir=/root/models \
#        --output_dir=/root/output_af3 \
#        >> "$LOG_FILE" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_SLOT apptainer exec --nv \
        --bind "$INPUT_DIR":/root/input_af3 \
        --bind "$OUTPUT_DIR":/root/output_af3 \
        --bind "$PARAM_DIR":/root/models \
        "$AF3_DIR/alphafold3_tacc.sif" \
        python3 $RUN_AF3_PATH \
        --buckets 5120 \
        --norun_data_pipeline \
        --json_path=/root/input_af3/${INPUT_JSON} \
        --model_dir=/root/models \
        --output_dir=/root/output_af3 \
        >> "$LOG_FILE" 2>&1

}

node_worker() {
    while true; do
        BATCH=($(claim_batch "$NGPUS"))
        if [ ${#BATCH[@]} -eq 0 ]; then
            echo "[$(hostname)] Cursor exhausted. Exiting."
            break
        fi
        slot=0
        for POOL_ID in "${BATCH[@]}"; do
            run_af3 "$POOL_ID" "$slot" &
            slot=$((slot + 1))
        done
        wait
    done
}

# Re-invoked per node by srun below; each node runs its own 4-way GPU loop
if [ "$1" == "--worker" ]; then
    node_worker
    exit 0
fi

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 "$0" --worker

