#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # bosch_cpu-cascadelake
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=autogluon_amlb_template
#SBATCH --export=ALL
#SBATCH --gres=gpu:1,localtmp:100
#SBATCH --propagate=NONE
#SBATCH -o x/slurm_out/slurm-%A_%a.out

set -e
set -u
set -o pipefail
set -x

# ===== Environment Settings
# Path to the Python executable of the AMLB venv
PYTHON=x/venv/bin/python
# Path to the runbenchmark.py script of the AMLB
RUNBENCHMARK=x/runbenchmark.py
# Path to the OpenML cache directory
DATACACHE=x/.openml-cache
# Path to the directory where the results will be stored
RESULT_DIR=x/results
# Path to the user directory of user dir for AMLB
USER_DIR=x/benchmarking-user-dir
# Path to the job check script
JOB_CHECK_SCRIPT=x/check_job_exists.py

# ===== Benchmark Settings
mapfile -t TASKS < "./tabrepo_amlb_tasks.txt"
FOLDS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
EXPERIMENTS=("Default" "New")
CONSTRAINTS=4h8c

# Calculate the total number of combinations
num_tasks=${#TASKS[@]}
num_folds=${#FOLDS[@]}
num_experiments=${#EXPERIMENTS[@]}
total_combinations=$((num_tasks * num_folds * num_experiments))

index=$SLURM_ARRAY_TASK_ID

# Ensure the index is within the valid range
if [ "$index" -lt 0 ] || [ "$index" -ge "$total_combinations" ]; then
    echo "Index out of range. Please provide an index between 0 and $((total_combinations - 1))."
    exit 1
fi

# Determine the combination based on the index
task_index=$((index / (num_experiments * num_folds)))
remaining=$((index % (num_experiments * num_folds)))
$experiment_index=$((remaining / num_folds))
fold_index=$((index % num_folds))

# Get the corresponding elements from the arrays
selected_task=${TASKS[$task_index]}
selected_fold=${FOLDS[$fold_index]}
selected_experiment=${EXPERIMENTS[$experiment_index]}
FRAMEWORK="AutoGluonLatest-${selected_experiment}"


outputString=$($PYTHON $JOB_CHECK_SCRIPT $selected_task $selected_fold $FRAMEWORK)
if [ "$outputString" == "True" ]; then
    echo "Job already exists. Exiting."
    exit 0
fi

# Output the selected combination
echo "Selected combination:"
echo "TASK: $selected_task"
echo "FOLD: $selected_fold"
echo "FRAMEWORK: $FRAMEWORK"

if [[ -z ${SLURM_ARRAY_TASK_ID} ]]; then
  echo "Please launch using:"
  echo "sbatch --array=0-${total_combinations} $0"
  exit
fi




$PYTHON $RUNBENCHMARK \
    $FRAMEWORK \
    "openml/t/$selected_task" \
    $CONSTRAINTS \
    --fold "$selected_fold" \
    --indir $DATACACHE \
    --outdir $RESULT_DIR \
    --userdir $USER_DIR \
    --setup "auto"