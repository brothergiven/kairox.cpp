#!/bin/bash
# main (A: 캐시 대비 실수 비율 스왑 예산 + tau 비활성) 에서 g=16 vs g=1 적중률 비교.
# dump_activation.sh 없이 llama-completion 을 직접 부른다. 인자는 이전 캠페인과 동일.
#
# usage: [RUNS=3] [N=512] [VB=6] [THREADS=4] [OUT_DIR=./hit_g1_vs_g16] bash run_hit_g1_vs_g16.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

RUNS=${RUNS:-3}
N=${N:-512}
VB=${VB:-6}
THREADS=${THREADS:-4}   # 8 코어 — ggml + 워커 1 + 메인 1 이 코어를 넘으면 안 된다
OUT_DIR=${OUT_DIR:-./hit_g1_vs_g16}
BIN=./build_rel/bin/llama-completion
MODEL=$HOME/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf
SPLIT_PREFIX=$HOME/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split
N_FF=11008

[[ -x $BIN ]] || { echo "빌드 없음 — bash compile_kairox.sh rel"; exit 1; }
mkdir -p "$OUT_DIR"
{
    git log --oneline -1
    git diff --quiet -- src ggml || echo "(src/ggml 에 커밋 안 된 변경 있음)"
    echo "RUNS=$RUNS N=$N VB=$VB THREADS=$THREADS"
} | tee "$OUT_DIR/meta.txt"

for g in 16 1; do
    csv=$OUT_DIR/g$g.csv
    log=$OUT_DIR/g$g.log
    if [[ -s $csv ]]; then
        echo "[skip] g=$g"
        continue
    fi

    echo "[run ] g=$g  $(date +%T)"
    # KAIROX_PARALLEL=1 필수: 없으면 RELOAD_PLAN 이 그래프에 안 들어가 캐시가 정적, 카운터 전부 0.
    # CSV 는 소멸자에서 쓰이므로 정상 종료해야 남는다.
    env CUDA_VISIBLE_DEVICES=0 \
        KAIROX_PARALLEL=1 \
        KAIROX_DUMP_ACTIVATION=1 KAIROX_DUMP_ACTIVATION_PATH="$csv" \
        KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05 \
    "$BIN" \
        -m "$MODEL" -kairox-ms "$SPLIT_PREFIX-$(( N_FF / g )).gguf" \
        -cffn -fit off -ngl all --no-mmap --no-direct-io \
        -vb "$VB" -t "$THREADS" \
        -no-cnv --repeat-penalty 1.1 --dry-multiplier 0.6 -s 42 \
        -c 1024 -n "$N" --ignore-eos --no-warmup \
        --bench-prompt-file prompts.txt --bench-runs "$RUNS" --bench-warmup 0 --bench-no-print \
        >"$log" 2>&1

    [[ -s $csv ]] || { echo "g=$g 실패 — $log 확인"; exit 1; }
done

# CSV: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
echo
printf "%-4s %9s %9s %14s\n" g hit/act wasted total_loads
for g in 16 1; do
    awk -F, -v g=$g 'NR>1 { a+=$3; h+=$5; t+=$6; w+=$7 }
        END { printf "%-4s %8.2f%% %8.2f%% %14d\n", g, h/a*100, (t ? w/t*100 : 0), t }' "$OUT_DIR/g$g.csv"
done
echo
echo "참고 — 원본 배포본 (정수 예산 + tau 0.33): g=16 59.48%  g=1 54.50%"
echo "     — 격리 조건 (예산 없음 + tau≈0)     : g=16 69.09%  g=1 85.63%"