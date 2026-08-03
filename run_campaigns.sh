#!/bin/bash
# run_campaigns.sh
# AD-MTDL optimization campaigns — sequential execution
# Seeds: moderate-zone compounds (pChEMBL 5.0–7.0), not Active for other targets
# This ensures the oracle has genuine optimization work to do.
#
# Usage:
#   bash run_campaigns.sh               # all 5 campaigns
#   bash run_campaigns.sh --only AChE   # single named campaign
#   bash run_campaigns.sh --from BACE1  # resume from BACE1 onward
#   bash run_campaigns.sh --dry_run     # print commands only

set -euo pipefail

# ══════════════════════════════════════════════════════════
# SEEDS — moderate-zone, selected by select_seeds.py
# Paste updated dual seed after rerunning select_seeds.py
# ══════════════════════════════════════════════════════════
SEED_ACHE="COc1ccc(C(=O)c2ccc(CN(C)Cc3ccccc3)cc2)cc1OC"
SEED_BUCHE="C=CCOc1ccc(CCNCc2cccc(OCc3ccccc3)c2)cc1"
SEED_BACE1="C=C1C[C@@](C)(c2cncc(NC(=O)c3ncc(C(F)(F)F)cc3Cl)c2)N=C(N)S1"
SEED_MAO_B="C#CCN(C)C1=C(Cl)C(=O)c2ccccc2C1=O"
SEED_ACHE_BUCHE="COc1cc(/C=N/NC(=O)N2CCN(c3ccccn3)CC2)cc(OC)c1OC"

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
STEPS=10000
BATCH_SIZE=64
TOP_K=1000
MIN_SCORE=0.40
N_CPUS=88
SERVER_PORT=8765
AD_MTDL_ENV="ad_mtdl"
RESULTS_BASE="campaigns"
CKPT="AD_MTDL_best.ckpt"
META="AD_MTDL_metadata.json"
PRIOR_SCAFFOLD="priors/mol2mol_medium_similarity.prior"
PRIOR_DENOVO="priors/reinvent.prior"

# ══════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════
ONLY_TARGET=""
FROM_TARGET=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)    ONLY_TARGET="$2"; shift 2 ;;
        --from)    FROM_TARGET="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true;     shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ══════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$RESULTS_BASE"
LOG_FILE="${RESULTS_BASE}/campaigns.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

run_or_dry() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    eval "$@"
    return $?
}

# ── Prerequisite checks ───────────────────────────────────
log "Checking prerequisites..."

for f in "$CKPT" "$META" "$PRIOR_SCAFFOLD" "$PRIOR_DENOVO" "optimize.py"; do
    if [[ ! -f "$f" ]]; then
        log "ERROR: Required file not found: $f"
        exit 1
    fi
done

if [[ "$SEED_ACHE_BUCHE" == "PASTE_UPDATED_DUAL_SEED_HERE" ]]; then
    log "ERROR: Dual seed not set. Rerun select_seeds.py and update SEED_ACHE_BUCHE."
    exit 1
fi

if ! conda run -n reinvent4 reinvent --version &>/dev/null; then
    log "ERROR: reinvent4 env not found or reinvent not working"
    exit 1
fi

log "All prerequisites OK"
log "Seeds:"
log "  AChE       : $SEED_ACHE"
log "  BuChE      : $SEED_BUCHE"
log "  BACE1      : $SEED_BACE1"
log "  MAO_B      : $SEED_MAO_B"
log "  AChE+BuChE : $SEED_ACHE_BUCHE"

# ══════════════════════════════════════════════════════════
# CAMPAIGN RUNNER
# ══════════════════════════════════════════════════════════
run_campaign() {
    local name="$1"
    local targets="$2"
    local seed="$3"
    local out_dir="${RESULTS_BASE}/${name}"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Campaign : $name"
    log "Targets  : $targets"
    log "Seed     : $seed"
    log "Output   : $out_dir"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Skip if already completed
    if [[ -f "${out_dir}/top_compounds.csv" ]]; then
        local n
        n=$(tail -n +2 "${out_dir}/top_compounds.csv" | wc -l)
        log "SKIP: already complete — ${n} compounds in ${out_dir}/top_compounds.csv"
        return 0
    fi

    local start_ts
    start_ts=$(date +%s)

    local cmd
    cmd="conda run --no-capture-output -n reinvent4 \
        python optimize.py \
        --smiles \"${seed}\" \
        --targets ${targets} \
        --mode scaffold_hop \
        --steps ${STEPS} \
        --batch_size ${BATCH_SIZE} \
        --top_k ${TOP_K} \
        --min_score ${MIN_SCORE} \
        --n_cpus ${N_CPUS} \
        --server_port ${SERVER_PORT} \
        --ad_mtdl_env ${AD_MTDL_ENV} \
        --ckpt ${CKPT} \
        --meta ${META} \
        --prior_scaffold ${PRIOR_SCAFFOLD} \
        --prior_denovo ${PRIOR_DENOVO} \
        --output ${out_dir} \
        2>&1 | tee -a ${LOG_FILE}"

    run_or_dry "$cmd"
    local exit_code=$?

    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$(( (end_ts - start_ts) / 60 ))

    if [[ $exit_code -eq 0 ]]; then
        log "Campaign $name DONE in ${elapsed} min"
        if [[ -f "${out_dir}/top_compounds.csv" ]]; then
            local n
            n=$(tail -n +2 "${out_dir}/top_compounds.csv" | wc -l)
            log "  → ${n} compounds saved to ${out_dir}/top_compounds.csv"
        fi
    else
        log "Campaign $name FAILED (exit $exit_code) after ${elapsed} min"
        log "  → Check: ${out_dir}/reinvent.log"
        log "  → Check: ${out_dir}/scoring_server.log"
    fi

    sleep 5   # allow OS to release file handles between campaigns
}

# ══════════════════════════════════════════════════════════
# CAMPAIGN DEFINITIONS
# ══════════════════════════════════════════════════════════
# Format: name | targets | seed
declare -A CAMPAIGN_TARGETS
declare -A CAMPAIGN_SEEDS

CAMPAIGN_TARGETS["1_AChE"]="AChE"
CAMPAIGN_SEEDS["1_AChE"]="$SEED_ACHE"

CAMPAIGN_TARGETS["2_BuChE"]="BuChE"
CAMPAIGN_SEEDS["2_BuChE"]="$SEED_BUCHE"

CAMPAIGN_TARGETS["3_BACE1"]="BACE1"
CAMPAIGN_SEEDS["3_BACE1"]="$SEED_BACE1"

CAMPAIGN_TARGETS["4_MAO_B"]="MAO_B"
CAMPAIGN_SEEDS["4_MAO_B"]="$SEED_MAO_B"

CAMPAIGN_TARGETS["5_AChE_BuChE"]="AChE BuChE"
CAMPAIGN_SEEDS["5_AChE_BuChE"]="$SEED_ACHE_BUCHE"

CAMPAIGN_ORDER=(
    "1_AChE"
    "2_BuChE"
    "3_BACE1"
    "4_MAO_B"
    "5_AChE_BuChE"
)

# ══════════════════════════════════════════════════════════
# DETERMINE RUN LIST
# ══════════════════════════════════════════════════════════
RUN_LIST=()
STARTED=false

for name in "${CAMPAIGN_ORDER[@]}"; do
    if [[ -n "$ONLY_TARGET" ]]; then
        [[ "$name" == *"$ONLY_TARGET"* ]] && RUN_LIST+=("$name")
        continue
    fi
    if [[ -n "$FROM_TARGET" && "$STARTED" == false ]]; then
        if [[ "$name" == *"$FROM_TARGET"* ]]; then
            STARTED=true
        else
            log "Skipping $name (before --from $FROM_TARGET)"
            continue
        fi
    fi
    RUN_LIST+=("$name")
done

if [[ ${#RUN_LIST[@]} -eq 0 ]]; then
    log "ERROR: No campaigns matched --only '$ONLY_TARGET' / --from '$FROM_TARGET'"
    exit 1
fi

# ══════════════════════════════════════════════════════════
# EXECUTE
# ══════════════════════════════════════════════════════════
TOTAL_START=$(date +%s)

log "═══════════════════════════════════════════════════"
log "AD-MTDL Campaign Run"
log "Campaigns : ${RUN_LIST[*]}"
log "Steps     : $STEPS per campaign"
log "Top K     : $TOP_K"
log "CPUs      : $N_CPUS"
log "Log       : $LOG_FILE"
log "═══════════════════════════════════════════════════"

PASSED=0
FAILED=0
FAILED_LIST=()

for name in "${RUN_LIST[@]}"; do
    if run_campaign \
        "$name" \
        "${CAMPAIGN_TARGETS[$name]}" \
        "${CAMPAIGN_SEEDS[$name]}"; then
        PASSED=$(( PASSED + 1 ))
    else
        FAILED=$(( FAILED + 1 ))
        FAILED_LIST+=("$name")
    fi
done

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))
TOTAL_HR=$(echo "scale=1; $TOTAL_MIN / 60" | bc)

log ""
log "═══════════════════════════════════════════════════"
log "ALL CAMPAIGNS COMPLETE"
log "Total time : ${TOTAL_MIN} min (${TOTAL_HR} hr)"
log "Passed     : $PASSED / ${#RUN_LIST[@]}"
log "Failed     : $FAILED"
[[ $FAILED -gt 0 ]] && log "Failed list: ${FAILED_LIST[*]}"
log ""
log "Results:"
for name in "${RUN_LIST[@]}"; do
    out_dir="${RESULTS_BASE}/${name}"
    if [[ -f "${out_dir}/top_compounds.csv" ]]; then
        n=$(tail -n +2 "${out_dir}/top_compounds.csv" | wc -l)
        log "  ✓ ${name:<20} ${n} compounds"
    else
        log "  ✗ ${name:<20} no output"
    fi
done
log "═══════════════════════════════════════════════════"
