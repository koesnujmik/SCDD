# export PYTHONNOUSERSITE=1
set -eu

# ============================================================================
# CIFAR-100 RLDD pipeline.
#
# Layout: experiments/${EXP_NAME}/{experts,initials,recovers,students}/
# Same artifact model as run.sh; defaults to import-expert + import-initial
# because the CIFAR-100 workflow typically reuses an existing baseline.
# ============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- per-run knobs ----------------------------------------------------------
EXP_NAME="baseline"
GPU_ID=0
TEACHER_ARCH_NAME="${TEACHER_ARCH_NAME:-${ARCH_NAME:-resnet32}}"
STUDENT_ARCH_NAME="${STUDENT_ARCH_NAME:-${TEACHER_ARCH_NAME}}"
IPC=10
SELECTION_METHOD="original"
IMBALANCE_RATE=0.1
SEED="${SEED:-42}"

# Optional: biased expert path for BDPC loss (passed only if non-empty)
BIASED_EXPERT_PATH="${REPO_ROOT}/"

# Source for expert import: best expert from the base experiment
BASE_BEST_EXPERT_CKPT="${REPO_ROOT}/experiments/cifar100_IF10/base/experts/expert_002/ckpt.best.pth.tar"

# expert artifact: train | import | reuse
EXPERT_MODE="${EXPERT_MODE:-train}"
EXPERT_SOURCE_PATH="${EXPERT_SOURCE_PATH:-${BASE_BEST_EXPERT_CKPT}}"   # required if mode=import
EXPERT_ID="${EXPERT_ID:-}"                                             # single-id compatibility for reuse
EXPERT_IDS="${EXPERT_IDS:-${EXPERT_ID}}"                               # whitespace-separated ids for reuse

# initial artifact: generate | import | reuse
INITIAL_MODE="${INITIAL_MODE:-generate}"
INITIAL_SOURCE_DIR="${INITIAL_SOURCE_DIR:-}"
INITIAL_ID="${INITIAL_ID:-}"

NUM_EXPERTS="${NUM_EXPERTS:-1}"     # train: count to train; reuse: count from EXPERT_IDS
NUM_INITIAL="${NUM_INITIAL:-1}"
NUM_RECOVERS="${NUM_RECOVERS:-5}"
NUM_STUDENTS="${NUM_STUDENTS:-1}"


is_supported_arch() {
    case "$1" in
        convnet|resnet18|resnet32|resnet34) return 0 ;;
        *) return 1 ;;
    esac
}

# import 모드는 외부에서 1개의 expert를 지정하므로 NUM_EXPERTS=1로 강제.
if [ "${EXPERT_MODE}" = "import" ] && [ "${NUM_EXPERTS}" != "1" ]; then
    echo "NUM_EXPERTS>1 only valid with EXPERT_MODE=train or reuse (current=${EXPERT_MODE})"; exit 1
fi

REUSE_EXPERT_IDS=()
if [ "${EXPERT_MODE}" = "reuse" ]; then
    read -r -a REUSE_EXPERT_IDS <<< "${EXPERT_IDS}"
    if [ "${#REUSE_EXPERT_IDS[@]}" -lt "${NUM_EXPERTS}" ]; then
        echo "EXPERT_MODE=reuse requires at least NUM_EXPERTS ids in EXPERT_IDS (NUM_EXPERTS=${NUM_EXPERTS}, EXPERT_IDS='${EXPERT_IDS}')"; exit 1
    fi
fi

WANDB_PROJECT='LTDD_cifar100'
WANDB_API_KEY="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"
export WANDB_MODE=online
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IF_VALUE=$(awk "BEGIN { printf \"%g\", 1/${IMBALANCE_RATE} }")
DATASET_TAG="cifar100_IF${IF_VALUE}"
EXP_ROOT="${REPO_ROOT}/experiments/${DATASET_TAG}/${EXP_NAME}"
REGISTRY="${EXP_ROOT}/registry.csv"
mkdir -p "${EXP_ROOT}"
source "${REPO_ROOT}/tools/exp_layout.sh"
ensure_registry "${REGISTRY}"

# =============================================================================
# Outer loop: NUM_EXPERTS expert artifacts.
# train mode allocates new experts; reuse mode uses the first NUM_EXPERTS ids in EXPERT_IDS.
# =============================================================================
for _e in $(seq 1 "${NUM_EXPERTS}"); do

# =============================================================================
# Stage 1: Expert artifact
# =============================================================================
case "${EXPERT_MODE}" in
  train)
    EXPERT_DIR=$(allocate_next "${EXP_ROOT}/experts" expert)
    EXPERT_ID=$(basename "${EXPERT_DIR}")
    register_artifact "${REGISTRY}" "${EXPERT_ID}" expert \
        exp_name="${EXP_NAME}" \
        expert_id="${EXPERT_ID}" expert_source_type=trained \
        dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
        imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
        notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
    (
        cd expert
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --dataset cifar100 -a "${TEACHER_ARCH_NAME}" --num_classes 100 \
            --imbanlance_rate $IMBALANCE_RATE --epochs 200 -b 64 --q 0.8 --gamma1 1 --gamma2 0.5 \
            --root_log "${EXP_ROOT}/experts" \
            --root_model "${EXP_ROOT}/experts" \
            --store_name "${EXPERT_ID}" \
            --seed "${SEED}" \
            --exp-out-subdir
    )
    EXPERT_CKPT="${EXPERT_DIR}/ckpt.best.pth.tar"
    write_metadata "${EXPERT_DIR}" expert trained \
        checkpoint_path="${EXPERT_CKPT}" \
        teacher_arch="${TEACHER_ARCH_NAME}" \
        student_arch="${STUDENT_ARCH_NAME}" \
        seed="${SEED}"
    update_artifact "${REGISTRY}" "${EXPERT_ID}" expert_ckpt="${EXPERT_CKPT}"
    ;;
  import)
    if [ -z "${EXPERT_SOURCE_PATH}" ]; then
        echo "EXPERT_MODE=import requires EXPERT_SOURCE_PATH"; exit 1
    fi
    EXPERT_DIR=$(allocate_next "${EXP_ROOT}/experts" baseline)
    EXPERT_ID=$(basename "${EXPERT_DIR}")
    cp "${EXPERT_SOURCE_PATH}" "${EXPERT_DIR}/ckpt.best.pth.tar"
    EXPERT_CKPT="${EXPERT_DIR}/ckpt.best.pth.tar"
    write_metadata "${EXPERT_DIR}" expert imported \
        source_path="${EXPERT_SOURCE_PATH}" \
        checkpoint_path="${EXPERT_CKPT}" \
        teacher_arch="${TEACHER_ARCH_NAME}" \
        student_arch="${STUDENT_ARCH_NAME}" \
        seed="${SEED}"
    register_artifact "${REGISTRY}" "${EXPERT_ID}" expert \
        exp_name="${EXP_NAME}" \
        expert_id="${EXPERT_ID}" expert_source_type=imported \
        expert_source_path="${EXPERT_SOURCE_PATH}" \
        expert_ckpt="${EXPERT_CKPT}" \
        dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
        imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
        notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
    update_artifact "${REGISTRY}" "${EXPERT_ID}" status=done
    ;;
  reuse)
    EXPERT_ID="${REUSE_EXPERT_IDS[$((_e - 1))]}"
    if [ -z "${EXPERT_ID}" ]; then
        echo "EXPERT_MODE=reuse requires EXPERT_ID"; exit 1
    fi
    EXPERT_DIR="${EXP_ROOT}/experts/${EXPERT_ID}"
    EXPERT_CKPT="${EXPERT_DIR}/ckpt.best.pth.tar"
    if [ ! -f "${EXPERT_CKPT}" ]; then
        echo "Expected expert ckpt not found: ${EXPERT_CKPT}"; exit 1
    fi
    ;;
  *)
    echo "Unknown EXPERT_MODE=${EXPERT_MODE}"; exit 1 ;;
esac

# =============================================================================
# Inner loop: NUM_INITIAL initial artifacts per expert.
# Each initial gets NUM_RECOVERS recovers, and each recover gets NUM_STUDENTS students.
# =============================================================================
for _i in $(seq 1 "${NUM_INITIAL}"); do

# =============================================================================
# Stage 2: Initial artifact
# =============================================================================
case "${INITIAL_MODE}" in
  generate)
    INITIAL_DIR=$(allocate_next "${EXP_ROOT}/initials" init)
    INITIAL_ID=$(basename "${INITIAL_DIR}")
    INITIAL_SYN_DIR="${INITIAL_DIR}/syn_data"
    register_artifact "${REGISTRY}" "${INITIAL_ID}" initial \
        exp_name="${EXP_NAME}" \
        expert_id="${EXPERT_ID}" \
        initial_id="${INITIAL_ID}" initial_source_type=generated \
        initial_dir="${INITIAL_SYN_DIR}" \
        dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
        imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
        notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
    (
        cd initial
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --subset "cifar100" \
            --imbanlance-rate $IMBALANCE_RATE \
            --arch-name "${TEACHER_ARCH_NAME}" \
            --factor 1 \
            --num-crop 1 \
            --mipc 500 \
            --ipc $IPC \
            --stud-name "${STUDENT_ARCH_NAME}" \
            --re-epochs 300 \
            --selection-method $SELECTION_METHOD \
            --pre-train-path "${EXPERT_CKPT}" \
            --syn-data-path "${INITIAL_SYN_DIR}" \
            --exp-name "${INITIAL_ID}" \
            --seed "${SEED}"
    )
    write_metadata "${INITIAL_DIR}" initial generated \
        parents.expert_id="${EXPERT_ID}" \
        syn_data_dir="${INITIAL_SYN_DIR}" \
        teacher_arch="${TEACHER_ARCH_NAME}" \
        student_arch="${STUDENT_ARCH_NAME}" \
        seed="${SEED}"
    update_artifact "${REGISTRY}" "${INITIAL_ID}"
    ;;
  import)
    if [ -z "${INITIAL_SOURCE_DIR}" ]; then
        echo "INITIAL_MODE=import requires INITIAL_SOURCE_DIR"; exit 1
    fi
    INITIAL_DIR=$(allocate_next "${EXP_ROOT}/initials" baseline_init)
    INITIAL_ID=$(basename "${INITIAL_DIR}")
    INITIAL_SYN_DIR="${INITIAL_DIR}/syn_data"
    cp -r "${INITIAL_SOURCE_DIR}" "${INITIAL_SYN_DIR}"
    write_metadata "${INITIAL_DIR}" initial imported \
        parents.expert_id="${EXPERT_ID}" \
        source_path="${INITIAL_SOURCE_DIR}" \
        syn_data_dir="${INITIAL_SYN_DIR}" \
        teacher_arch="${TEACHER_ARCH_NAME}" \
        student_arch="${STUDENT_ARCH_NAME}" \
        seed="${SEED}"
    register_artifact "${REGISTRY}" "${INITIAL_ID}" initial \
        exp_name="${EXP_NAME}" \
        expert_id="${EXPERT_ID}" \
        initial_id="${INITIAL_ID}" initial_source_type=imported \
        initial_source_path="${INITIAL_SOURCE_DIR}" \
        initial_dir="${INITIAL_SYN_DIR}" \
        dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
        imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
        notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
    update_artifact "${REGISTRY}" "${INITIAL_ID}" status=done
    ;;
  reuse)
    if [ -z "${INITIAL_ID}" ]; then
        echo "INITIAL_MODE=reuse requires INITIAL_ID"; exit 1
    fi
    INITIAL_DIR="${EXP_ROOT}/initials/${INITIAL_ID}"
    INITIAL_SYN_DIR="${INITIAL_DIR}/syn_data"
    if [ ! -d "${INITIAL_SYN_DIR}" ]; then
        echo "Expected initial syn_data not found: ${INITIAL_SYN_DIR}"; exit 1
    fi
    ;;
  *)
    echo "Unknown INITIAL_MODE=${INITIAL_MODE}"; exit 1 ;;
esac

# =============================================================================
# Stage 3 + 4: NUM_RECOVERS recovers per initial, each with NUM_STUDENTS students
# =============================================================================
for _r in $(seq 1 "${NUM_RECOVERS}"); do
    RECOVER_DIR=$(allocate_next "${EXP_ROOT}/recovers" recover)
    RECOVER_ID=$(basename "${RECOVER_DIR}")
    RECOVER_SYN_DIR="${RECOVER_DIR}/syn_data"
    register_artifact "${REGISTRY}" "${RECOVER_ID}" recover \
        exp_name="${EXP_NAME}" \
        expert_id="${EXPERT_ID}" initial_id="${INITIAL_ID}" \
        recover_id="${RECOVER_ID}" \
        recover_dir="${RECOVER_DIR}" recover_syn_dir="${RECOVER_SYN_DIR}" \
        dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
        imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
        notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
    (
        cd recover_cifar100
        CUDA_VISIBLE_DEVICES=$GPU_ID python recover.py \
            --arch-name "${TEACHER_ARCH_NAME}" \
            --exp-name "${RECOVER_ID}" \
            --batch-size 100 --category-aware "global" \
            --lr 0.05 --drop-rate 0.0 \
            --ipc-number $IPC --training-momentum 0.8 \
            --iteration 2000 \
            --imbanlance-rate $IMBALANCE_RATE \
            --r-loss 0.01 \
            --verifier --store-best-images --gpu-id $GPU_ID \
            --pre-train-path "${EXPERT_CKPT}" \
            --initial-img-dir "${INITIAL_SYN_DIR}" \
            --syn-data-path "${RECOVER_SYN_DIR}" \
            --statistic-path "${EXPERT_DIR}/statistic" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-api-key "${WANDB_API_KEY}" \
            --wandb-run-name "${EXP_NAME}_e${_e}_i${_i}_r${_r}" \
            --seed "${SEED}"
    )
    write_metadata "${RECOVER_DIR}" recover generated \
        parents.expert_id="${EXPERT_ID}" \
        parents.initial_id="${INITIAL_ID}" \
        syn_data_dir="${RECOVER_SYN_DIR}" \
        teacher_arch="${TEACHER_ARCH_NAME}" \
        student_arch="${STUDENT_ARCH_NAME}" \
        seed="${SEED}"
    WANDB_RUN_ID_FILE="${RECOVER_SYN_DIR}/wandb_run_id.txt"
    WANDB_RUN_ID=""
    [ -f "${WANDB_RUN_ID_FILE}" ] && WANDB_RUN_ID=$(cat "${WANDB_RUN_ID_FILE}")
    update_artifact "${REGISTRY}" "${RECOVER_ID}" wandb_run_id="${WANDB_RUN_ID}"

    for _s in $(seq 1 "${NUM_STUDENTS}"); do
        STUDENT_DIR=$(allocate_next "${EXP_ROOT}/students" student)
        STUDENT_ID=$(basename "${STUDENT_DIR}")
        register_artifact "${REGISTRY}" "${STUDENT_ID}" student \
            exp_name="${EXP_NAME}" \
            expert_id="${EXPERT_ID}" initial_id="${INITIAL_ID}" \
            recover_id="${RECOVER_ID}" \
            student_id="${STUDENT_ID}" student_output_dir="${STUDENT_DIR}" \
            dataset=cifar100 arch="${TEACHER_ARCH_NAME}" ipc="${IPC}" \
            imbalance_rate="${IMBALANCE_RATE}" seed="${SEED}" \
            notes="teacher_arch=${TEACHER_ARCH_NAME};student_arch=${STUDENT_ARCH_NAME}"
        (
            cd train_cifar100
            CUDA_VISIBLE_DEVICES=$GPU_ID python direct_train.py \
                --wandb-project "${WANDB_PROJECT}" \
                --wandb-api-key "${WANDB_API_KEY}" \
                --wandb-run-name "${EXP_NAME}_e${_e}_i${_i}_r${_r}_s${_s}" \
                --batch-size 100 --epochs 1000 \
                --model "${STUDENT_ARCH_NAME}" \
                --teacher-arch "${TEACHER_ARCH_NAME}" \
                --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
                -T 20 --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id $GPU_ID \
                -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
                --mix-type 'cutmix' --adamw-weight-decay 0.0005 \
                --output-dir "${STUDENT_DIR}/" \
                --train-dir "${RECOVER_SYN_DIR}" \
                --pre-train-path "${EXPERT_CKPT}" \
                --seed "${SEED}"
        )
        write_metadata "${STUDENT_DIR}" student trained \
            parents.recover_id="${RECOVER_ID}" \
            parents.expert_id="${EXPERT_ID}" \
            parents.initial_id="${INITIAL_ID}" \
            teacher_arch="${TEACHER_ARCH_NAME}" \
            student_arch="${STUDENT_ARCH_NAME}" \
            seed="${SEED}"
        BEST_ACC=""
        WID=""
        if [ -f "${STUDENT_DIR}/summary.json" ]; then
            BEST_ACC=$(python3 -c "import json; print(json.load(open('${STUDENT_DIR}/summary.json'))['best_acc1'])" 2>/dev/null || echo "")
            WID=$(python3 -c "import json; v=json.load(open('${STUDENT_DIR}/summary.json')).get('wandb_run_id'); print(v if v else '')" 2>/dev/null || echo "")
        fi
        update_artifact "${REGISTRY}" "${STUDENT_ID}" \
            best_acc1="${BEST_ACC}" \
            wandb_run_id="${WID}"
    done
done

done   # end of NUM_INITIAL inner loop

unset EXPERT_ID INITIAL_ID

done   # end of NUM_EXPERTS outer loop

python3 "${REPO_ROOT}/tools/summarize.py" "${EXP_ROOT}" || true
echo "Done. Registry: ${REGISTRY}"
