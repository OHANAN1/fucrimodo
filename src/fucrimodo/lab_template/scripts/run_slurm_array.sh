#!/bin/bash
# ╒════════════════════════════════════════════════════════════════════════════╕
#       Slurm array job: run fucrimodo over a directory of input files
# ╘════════════════════════════════════════════════════════════════════════════╛
#
# Purpose:
#   For each input file in <input_dir>, launch one fucrimodo run and save its
#   output to <save_dir>/<run_name>.
#
# Quick start:
#   1. Edit the "USER CONFIGURATION" section below.
#   2. Submit with: sbatch run_fucrimodo_array.sh
#   3. Realize you should have first read the documentation.
#
# Notes:
#   - The number of array tasks (0-99 here) must match the number of files in
#     <input_dir>.
#   - Each task uses 4 CPU threads; set OMP/MKL/etc. accordingly.
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=n-atoms-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=physik
#SBATCH --exclude=slurm-exec-lm-002
#SBATCH --array=0-99
#SBATCH --output=output_%A_%a.txt

# ─────────────────────────────────────────────────────────────────────────────
#                          USER CONFIGURATION
#    Edit the variables below before submitting. Also, please do not commit
#             secrets or personal paths to a shared repository.
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: Depending on the number of files adjust the `#SBATCH --array=0-99`
# e.g. 1000 files: `#SBATCH --array=0-999`

# Slurm account to use (replace with your own, e.g. "fujimoto")
SLURM_ACCOUNT="fujimoto"

# Absolute path to the conda executable you want to use.
# This assumes fucrimodo is installed with conda!
# To find your path, run: `which conda`
# Then use the absolute path to .../bin/activate.
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

# Define paths
repo_dir="$PWD"                                  # root of fucrimodo lab
save_dir="${repo_dir}/data/results/slurm-run"    # path to store results
input_dir="${repo_dir}/data/raw/multi-run/"      # dir with multiple target files
config_file="${repo_dir}/configs/run/default.py" # path to run config

# ─────────────────────────────────────────────────────────────────────────────
#                     END OF USER CONFIGURATION
#              No need to change anything below this line.
#                   You should still read it, tho.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#              Validation: make sure user edited the config
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${SLURM_ACCOUNT}" == "REPLACE_WITH_YOUR_ACCOUNT" || -z "${SLURM_ACCOUNT}" ]]; then
	echo "ERROR: Please set SLURM_ACCOUNT in the USER CONFIGURATION section." >&2
	exit 1
fi

if [[ ! -f "${CONDA_ACTIVATE}" ]]; then
	echo "ERROR: Conda activate script not found: '${CONDA_ACTIVATE}'" >&2
	echo "       Please set CONDA_ACTIVATE to the correct path." >&2
	exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
#                             Logging helper
# ─────────────────────────────────────────────────────────────────────────────
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ─────────────────────────────────────────────────────────────────────────────
#                           Load required modules
# ─────────────────────────────────────────────────────────────────────────────
log "Loading git module..."
module load git

# ─────────────────────────────────────────────────────────────────────────────
#                       Activate the conda environment
# ─────────────────────────────────────────────────────────────────────────────
log "Activating conda environment 'fucrimodo-env'..."
source "$CONDA_ACTIVATE" fucrimodo-env

# ─────────────────────────────────────────────────────────────────────────────
#                       Limit parallel executions
# ─────────────────────────────────────────────────────────────────────────────
# This is for libs like numpy, scipy, ...
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ─────────────────────────────────────────────────────────────────────────────
#                 Ensure we run from the fucrimodo_lab
# ─────────────────────────────────────────────────────────────────────────────
cd "${repo_dir}" || {
	log "ERROR: Could not change to repository directory '${repo_dir}'"
	exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
#           Create the output directory if it does not exist
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "${save_dir}"

# ─────────────────────────────────────────────────────────────────────────────
#                Print Slurm job metadata for the logs
# ─────────────────────────────────────────────────────────────────────────────
log "Job info:"
log "  SLURM_JOB_ID        = ${SLURM_JOB_ID:-N/A}"
log "  SLURM_ARRAY_JOB_ID  = ${SLURM_ARRAY_JOB_ID:-N/A}"
log "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID:-N/A}"
log "  Host                = $(hostname)"
log "  Working directory   = ${PWD}"

# ─────────────────────────────────────────────────────────────────────────────
#            Map array task ID to a deterministic input file
# ─────────────────────────────────────────────────────────────────────────────
# Build a sorted list of input files. Sorting guarantees that the same task ID
# always maps to the same file, even if the filesystem order changes.
mapfile -t input_files < <(find "${input_dir}" -maxdepth 1 -type f -printf '%f\n' | sort)

num_files=${#input_files[@]}
task_id=${SLURM_ARRAY_TASK_ID}

log "Found ${num_files} input files in '${input_dir}'"

# Validate that the task ID is within range
if ((task_id >= num_files)); then
	log "ERROR: SLURM_ARRAY_TASK_ID (${task_id}) is out of range (0..$((num_files - 1)))"
	exit 1
fi

input_file_name="${input_files[$task_id]}"
input_file_path="${input_dir}/${input_file_name}"

# Verify the selected file exists and is readable
if [[ ! -r "${input_file_path}" ]]; then
	log "ERROR: Input file is not readable: '${input_file_path}'"
	exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
#       Derive the run name from the input file name (without extension)
# ─────────────────────────────────────────────────────────────────────────────
# Example: my_target.json -> my_target
run_name="${input_file_name%.*}"

log "Selected input file: '${input_file_path}'"
log "Run name: '${run_name}'"

# ─────────────────────────────────────────────────────────────────────────────
#                              Run fucrimodo
# ─────────────────────────────────────────────────────────────────────────────
log "Starting fucrimodo..."

fucrimodo -T run \
	-c "${config_file}" \
	-n "${run_name}" \
	-s "${save_dir}" \
	-p "${SLURM_CPUS_PER_TASK}" \
	"${input_file_path}"

log "fucrimodo finished successfully"

#  You really read the script!
# Here is a star for the effort!
#
#             .
#        ---./|\.---
#        '._/ | \_.'
#      _.-'_'.|.'_'-._
#       '-._.'|'._.-'
#        .' \ | / '.
#        ---'\|/'---
#             ' Starshine
#
#  (`Ponyo wants ham!`~Ponyo)
