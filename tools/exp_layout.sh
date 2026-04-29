# tools/exp_layout.sh
# Helpers for the experiments/{exp_name}/ artifact layout.
# Source from run.sh:  source tools/exp_layout.sh
#
# Public functions:
#   allocate_next      <parent_dir> <prefix>
#   write_metadata     <artifact_dir> <artifact_type> <source_type> [extra json fields...]
#   ensure_registry    <csv_path>
#   register_artifact  <csv_path> <run_id> <stage> <kvpair>...
#   update_artifact    <csv_path> <run_id> <kvpair>...
#
# CSV columns (kept in this exact order):
_REG_COLS=(
    run_id exp_name stage status
    expert_id expert_source_type expert_source_path expert_ckpt
    initial_id initial_source_type initial_source_path initial_dir
    recover_id recover_dir recover_syn_dir
    student_id student_output_dir
    dataset arch ipc imbalance_rate
    best_acc1 wandb_run_id seed started_at ended_at git_commit notes
)

# allocate_next <parent_dir> <prefix>
# Atomically creates "<parent>/<prefix>_NNN" with NNN = max(existing)+1.
# Echoes the absolute path of the new directory.
# Safe under concurrent callers: mkdir is atomic on POSIX.
allocate_next() {
    local parent="$1" prefix="$2"
    mkdir -p "$parent"
    while true; do
        local max
        max=$(ls "$parent" 2>/dev/null \
            | grep -E "^${prefix}_[0-9]+$" \
            | sed "s/^${prefix}_//" \
            | sort -n | tail -1)
        local next
        next=$(printf "%03d" $(( 10#${max:-0} + 1 )))
        local target="$parent/${prefix}_${next}"
        if mkdir "$target" 2>/dev/null; then
            (cd "$target" && pwd)
            return 0
        fi
        # collision: another runner grabbed this id, retry
    done
}

_now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

_git_commit() {
    git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." rev-parse --short HEAD 2>/dev/null || echo ""
}

# write_metadata <artifact_dir> <artifact_type> <source_type> [k=v ...]
# Writes <artifact_dir>/metadata.json. Extra k=v pairs become top-level fields
# (values are JSON-quoted as strings; nest under parents.* with prefix "parents.").
write_metadata() {
    local dir="$1" atype="$2" stype="$3"
    shift 3
    local artifact_id
    artifact_id=$(basename "$dir")
    local tmp="$dir/metadata.json.tmp"
    {
        printf '{\n'
        printf '  "artifact_id": "%s",\n' "$artifact_id"
        printf '  "artifact_type": "%s",\n' "$atype"
        printf '  "source_type": "%s",\n' "$stype"
        printf '  "created_at": "%s",\n' "$(_now_utc)"
        printf '  "git_commit": "%s"' "$(_git_commit)"
        local parents_open=0 first_parent=1
        for kv in "$@"; do
            local k="${kv%%=*}" v="${kv#*=}"
            case "$k" in
                parents.*)
                    if [ $parents_open -eq 0 ]; then
                        printf ',\n  "parents": {\n'
                        parents_open=1
                    fi
                    if [ $first_parent -eq 0 ]; then printf ',\n'; fi
                    printf '    "%s": "%s"' "${k#parents.}" "$v"
                    first_parent=0
                    ;;
                *)
                    if [ $parents_open -eq 1 ]; then
                        printf '\n  }'
                        parents_open=0
                    fi
                    printf ',\n  "%s": "%s"' "$k" "$v"
                    ;;
            esac
        done
        if [ $parents_open -eq 1 ]; then printf '\n  }'; fi
        printf '\n}\n'
    } > "$tmp"
    mv "$tmp" "$dir/metadata.json"
}

# ensure_registry <csv_path>
ensure_registry() {
    local csv="$1"
    if [ ! -f "$csv" ]; then
        mkdir -p "$(dirname "$csv")"
        local IFS=,
        printf '%s\n' "${_REG_COLS[*]}" > "$csv"
    fi
}

# Internal: read existing row by run_id into an array of "k=v" pairs.
# Echoes nothing if not found. Sets $? to 0 if found, 1 otherwise.
_read_row() {
    local csv="$1" run_id="$2"
    awk -F, -v rid="$run_id" 'NR==1{for(i=1;i<=NF;i++)h[i]=$i; next}
        $1==rid {for(i=1;i<=NF;i++) printf "%s=%s\n", h[i], $i; exit}' "$csv"
}

# register_artifact <csv_path> <run_id> <stage> <k=v>...
# Appends a new row with status=running and started_at=now.
# kvpairs override defaults; unspecified columns become "".
register_artifact() {
    local csv="$1" run_id="$2" stage="$3"
    shift 3
    ensure_registry "$csv"
    declare -A row
    row[run_id]="$run_id"
    row[stage]="$stage"
    row[status]="running"
    row[started_at]="$(_now_utc)"
    row[git_commit]="$(_git_commit)"
    for kv in "$@"; do
        local _v="${kv#*=}"
        # CSV stays comma-delimited; replace any commas/newlines/quotes in values
        # with safe substitutes so simple awk -F, parsing remains valid.
        _v="${_v//,/;}"; _v="${_v//$'\n'/ }"; _v="${_v//\"/\'}"
        row["${kv%%=*}"]="$_v"
    done
    (
        flock -x 9
        local out=""
        local first=1
        for col in "${_REG_COLS[@]}"; do
            local v="${row[$col]:-}"
            if [ $first -eq 1 ]; then out="$v"; first=0
            else out="$out,$v"; fi
        done
        printf '%s\n' "$out" >> "$csv"
    ) 9>"$csv.lock"
}

# update_artifact <csv_path> <run_id> <k=v>...
# In-place updates the row matching run_id. status defaults to "done" if not given.
# Sets ended_at=now if status transitions to done.
update_artifact() {
    local csv="$1" run_id="$2"
    shift 2
    ensure_registry "$csv"
    declare -A patch
    for kv in "$@"; do
        local _v="${kv#*=}"
        _v="${_v//,/;}"; _v="${_v//$'\n'/ }"; _v="${_v//\"/\'}"
        patch["${kv%%=*}"]="$_v"
    done
    if [ -z "${patch[status]:-}" ]; then patch[status]="done"; fi
    if [ "${patch[status]}" = "done" ] && [ -z "${patch[ended_at]:-}" ]; then
        patch[ended_at]="$(_now_utc)"
    fi
    (
        flock -x 9
        local tmp="$csv.tmp.$$"
        # Build awk assignments: col_index -> CSV-escaped value
        local awk_assigns=""
        local idx=1
        for col in "${_REG_COLS[@]}"; do
            if [ -n "${patch[$col]+x}" ]; then
                local v="${patch[$col]}"
                # awk-escape: backslash and double-quote
                v=${v//\\/\\\\}
                v=${v//\"/\\\"}
                awk_assigns+="\$$idx=\"$v\"; "
            fi
            idx=$((idx+1))
        done
        awk -F, -v OFS=, -v rid="$run_id" "
            NR==1 { print; next }
            \$1==rid { $awk_assigns print; next }
            { print }
        " "$csv" > "$tmp"
        mv "$tmp" "$csv"
    ) 9>"$csv.lock"
}
