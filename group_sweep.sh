#!/bin/bash

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

model_dir=${MODEL_DIR:-$HOME/SPIF-GGUF}
sizes=${SIZES:-"4 8 16 32 64"}
out_dir=${OUT_DIR:-$repo_root/dumps}
n_neurons=11008   # prosparse-llama-2-7b 의 n_ff

die() { echo "error: $*" >&2; exit 1; }

mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패: $out_dir"

# 스윕 실행 -------------------------------------------------------------------
for gs in $sizes; do
    n_group=$((n_neurons / gs))
    split=$model_dir/prosparse-llama-2-7b-sparkinfer-model-split-${n_group}.gguf # split 된 모델
    csv=$out_dir/gs${gs}.csv

    if [[ ! -f "$split" ]]; then
        echo "skip group_size=$gs — model-split 없음: $split" >&2
        echo "  생성: python3 regroup_model_split.py <원본-688.gguf> $gs" >&2
        continue
    fi

    echo
    echo "=============================================================="
    echo " group_size=$gs  (n_group=$n_group)"
    echo "=============================================================="

    # 런마다 생성 길이가 달라지면 토큰 스텝 수가 달라져 지표 비교가 오염된다(README 5-6).
    MODEL_SPLIT="$split" OUT="$csv" IGNORE_EOS="${IGNORE_EOS:-1}" bash dump_activation.sh || {
        echo "warning: group_size=$gs 실행 실패 — 건너뜀" >&2
        continue
    }
done

# 비교표 ----------------------------------------------------------------------
echo
echo "=============================================================="
echo " 스윕 요약"
echo "=============================================================="
printf "%-6s %10s %10s %12s %12s\n" gs hit/res hit/act wasted/total loads
for gs in $sizes; do
    csv=$out_dir/gs${gs}.csv
    [[ -s "$csv" ]] || continue
    awk -F, -v gs="$gs" 'NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
        END {
            printf "%-6s %9.2f%% %9.2f%% %11.2f%% %12d\n",
                gs, (r?h/r*100:0), (a?h/a*100:0), (t?w/t*100:0), t
        }' "$csv"
done

echo
echo "레이어별 hit/act 비교"
printf "%-6s" layer
for gs in $sizes; do [[ -s "$out_dir/gs${gs}.csv" ]] && printf "%10s" "gs=$gs"; done
echo
for l in $(seq 0 31); do
    printf "L%-5d" "$l"
    for gs in $sizes; do
        csv=$out_dir/gs${gs}.csv
        [[ -s "$csv" ]] || continue
        awk -F, -v L="$l" 'NR>1 && $1==L { a+=$3; h+=$5 }
            END { printf "%9.2f%%", (a?h/a*100:0) }' "$csv"
    done
    echo
done

echo
echo "csv: $out_dir/"
