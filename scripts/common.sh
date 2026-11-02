# sourced by the job scripts (run them from the repo root)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aml_tgcn
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_CBWR=AUTO
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs
