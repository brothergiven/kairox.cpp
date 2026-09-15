#!/bin/bash
# 전송 병합(KAIROX_GATHER) on/off 의 e2e decode 처리량 비교: g x gather, 라운드 순환.
#
# usage: [SIZES="16 1"] [ROUNDS=3] [RUNS=3] [N=512] [VB=6] [THREADS=4] [OUT_DIR=./e2e_gather] bash run_e2e_gather.sh
# 로그에 "decode mean" 이 있는 런은 건너뛰므로 중단 후 다시 실행하면 이어서 돈다.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

SIZES=${SIZES:-"16 1"}
ROUNDS=${ROUNDS:-3}
RUNS=${RUNS:-3}
N=${N:-512}
VB=${VB:-6}
THREADS=${THREADS:-4}
OUT_DIR=${OUT_DIR:-./e2e_gather}
BIN=./build_rel/bin/llama-completion
MODEL=$HOME/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf
SPLIT_PREFIX=$HOME/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split
N_FF=11008

die() { echo "error: $*" >&2; exit 1; }

[[ -x $BIN ]] || die "빌드 없음 — bash compile_kairox.sh rel"
# ggml 스레드풀은 스핀 대기 — ggml + 워커 1 + 메인 1 이 코어를 넘으면 처리량이 반토막 난다.
(( THREADS + 2 <= $(nproc) )) || die "THREADS=$THREADS 과다 (THREADS+2 <= nproc=$(nproc))"
# 다른 GPU 작업과 겹치면 시간이 통째로 오염된다. 절대 남의 프로세스를 죽이지 않는다.
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] ||
    die "GPU 를 쓰는 프로세스가 있다 — 끝난 뒤 다시 실행"

mkdir -p "$OUT_DIR"
{
    git log --oneline -1
    git diff --quiet -- src ggml || echo "(src/ggml 에 커밋 안 된 변경 있음)"
    echo "SIZES='$SIZES' ROUNDS=$ROUNDS RUNS=$RUNS N=$N VB=$VB THREADS=$THREADS"
} | tee "$OUT_DIR/meta.txt"

pswpin() { awk '/^pswpin/ { print $2 }' /proc/vmstat; }

run_one() {
    local g=$1 gather=$2 r=$3
    local tag="g${g}_gather${gather}_r${r}"
    local log="$OUT_DIR/$tag.log"

    if grep -q "decode mean" "$log" 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi

    # 호스트 RAM 이 빠듯하면 연속 실행이 swap 으로 무너진다 (같은 설정에서 15.5 / 0.6 t/s 가 섞인 적 있음).
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

    echo "[run ] $tag  $(date +%T)"
    local sw0; sw0=$(pswpin)

    # 처리량 측정이므로 activation 덤프는 끈다 (레이어마다 D2H + 동기화가 추가된다).
    env CUDA_VISIBLE_DEVICES=0 \
        KAIROX_PARALLEL=1 \
        KAIROX_GATHER="$gather" \
        KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05 \
    "$BIN" \
        -m "$MODEL" -kairox-ms "$SPLIT_PREFIX-$(( N_FF / g )).gguf" \
        -cffn -fit off -ngl all --no-mmap --no-direct-io \
        -vb "$VB" -t "$THREADS" \
        -no-cnv --repeat-penalty 1.1 --dry-multiplier 0.6 -s 42 \
        -c 1024 -n "$N" --ignore-eos --no-warmup \
        --bench-prompt-file prompts.txt --bench-runs "$RUNS" --bench-warmup 0 --bench-no-print \
        >"$log" 2>&1 || echo "  !! $tag 실패 — $log"

    local sw=$(( $(pswpin) - sw0 ))
    echo "pswpin_delta=$sw" >>"$log"
    (( sw > 1000 )) && echo "  !! swap +$sw — 요약에서 제외된다"
}

# 연속 반복하면 뒤로 갈수록 빨라지는 드리프트가 있다. 라운드로 순환하고, 라운드마다 on/off 순서를 뒤집는다.
for r in $(seq 1 "$ROUNDS"); do
    for g in $SIZES; do
        if (( r % 2 )); then order="0 1"; else order="1 0"; fi
        for gather in $order; do
            run_one "$g" "$gather" "$r"
        done
    done
done

# --- 요약 -------------------------------------------------------------------
# 로그 줄: bench run attempt 3 (included 3/3): prompt = 370 tok, decode = 511 tok, prefill = .. t/s, decode = 14.89 t/s
for f in "$OUT_DIR"/g*_gather*_r*.log; do
    sw=$(awk -F= '/^pswpin_delta=/ { print $2 }' "$f")
    (( ${sw:-0} > 1000 )) && continue
    awk -v tag="$(basename "$f" .log)" '
        match($0, /included [0-9]+/) {
            k = substr($0, RSTART + 9, RLENGTH - 9)
            if (match($0, /decode = [0-9.]+ t\/s/)) print tag, k, substr($0, RSTART + 9, RLENGTH - 13)
        }' "$f"
done >"$OUT_DIR/per_prompt.txt"

echo
printf "%-4s %-7s %4s %9s %9s\n" g gather n "t/s" "±95%"
awk '{
        split($1, a, "_"); g = substr(a[1], 2); on = substr(a[2], 7); r = substr(a[3], 2)
        key = g SUBSEP on; n[key]++; s[key] += $3; ss[key] += $3 * $3
        val[g, on, r, $2] = $3; gs[g] = 1; rs[r] = 1; ks[$2] = 1
    }
    END {
        for (g in gs) {
            for (on = 0; on <= 1; on++) {
                key = g SUBSEP on
                if (!n[key]) continue
                m  = s[key] / n[key]
                sd = (n[key] > 1) ? sqrt((ss[key] - n[key] * m * m) / (n[key] - 1)) : 0
                printf "%-4s %-7s %4d %9.2f %9.2f\n", g, (on ? "on" : "off"), n[key], m, 1.96 * sd / sqrt(n[key])
                mean[g, on] = m
            }
            # 같은 라운드·같은 프롬프트끼리 짝지어 비율을 본다 — 프롬프트 간 편차가 상쇄돼 평균비보다 안정적이다.
            np = 0; lr = 0
            for (r in rs) for (k in ks)
                if (((g, 0, r, k) in val) && ((g, 1, r, k) in val)) { lr += log(val[g, 1, r, k] / val[g, 0, r, k]); np++ }
            if (np) printf "     -> gather on/off: 평균비 %.2fx, 짝지은 기하평균 %.2fx (쌍 %d)\n", mean[g, 1] / mean[g, 0], exp(lr / np), np
        }
    }' "$OUT_DIR/per_prompt.txt"