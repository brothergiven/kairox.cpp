#!/bin/bash
# 배치 크기(np) x (g, gather) 조건의 e2e 처리량 (llama-batched).
#
# usage: [NPS="1 2 4 8"] [PAIRS="16:0 16:1 1:1"] [ROUNDS=5] [N=256] [CTX=2048]
#        [VB=6] [THREADS=4] [OUT_DIR=./batch_e2e] bash exp2_run_batch.sh
#
# 조건 (PAIRS 의 g:gather)
#   16:0  base  원본 전송 (그룹당 memcpy)
#   16:1  B1    전송 병합만
#   1:1   ours  전송 병합 + 뉴런 단위 입도
#   g=1 + gather off 는 전송 호출이 16 배라 느리기만 하고 비교에 필요 없어 기본에서 뺐다.
#   base 는 main 의 비율 예산(A) + tau 비활성 상태의 g=16 이다 — 정수 예산 / tau=0.33 원본이 아니다.
#
# CTX 는 모든 np 에 같은 값을 쓴다 — KV 캐시 크기가 np 마다 달라지면 KAIROX 캐시 용량이 달라져
# 배치 효과와 용량 효과가 섞인다. 가장 큰 np 가 들어가는 값으로 잡을 것:
#   필요 KV 토큰 = 프롬프트 + (N - 프롬프트) * np        (batched 의 -n 은 프롬프트 포함 전체 길이)
# f16 KV 로 np=8 까지 같은 캐시(약 1.36 GiB)를 유지하려면 N=256, CTX=2048.
# 전제: batched.cpp 가 -c 를 하한으로 존중하도록 수정돼 있어야 한다 (첫 런에서 검사한다).
#
# 주의: 실행 중에 이 파일을 수정하지 말 것. bash 는 실행하면서 파일을 읽어 들여 도중 수정 시 엉뚱한 줄을 실행한다.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

NPS=${NPS:-"1 2 4 8"}
PAIRS=${PAIRS:-"16:0 16:1 1:1"}
ROUNDS=${ROUNDS:-5}
N=${N:-256}
CTX=${CTX:-2048}
VB=${VB:-6}
THREADS=${THREADS:-4}
OUT_DIR=${OUT_DIR:-./batch_e2e}
PROMPT=${PROMPT:-"Explain how operating systems manage virtual memory, including paging, page tables, TLBs, and what happens on a page fault."}
BIN=./build_rel/bin/llama-batched
MODEL=$HOME/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf
SPLIT_PREFIX=$HOME/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split
N_FF=11008

die() { echo "error: $*" >&2; exit 1; }

# --- 사전 점검 ---------------------------------------------------------------
[[ -x $BIN ]] || die "llama-batched 없음 — cmake --build build_rel --config Release --target llama-batched"
# ggml 스레드풀은 스핀 대기한다. ggml + KAIROX 워커 1 + 메인 1 이 코어를 넘으면 처리량이 반토막 난다.
(( THREADS + 2 <= $(nproc) )) || die "THREADS=$THREADS 과다 (THREADS+2 <= nproc=$(nproc))"
# 다른 GPU 작업과 겹치면 시간이 통째로 오염된다. 절대 남의 프로세스를 죽이지 않는다.
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] ||
    die "GPU 를 쓰는 프로세스가 있다 — 끝난 뒤 다시 실행"

for p in $PAIRS; do
    [[ $p =~ ^[0-9]+:[01]$ ]] || die "PAIRS 형식 오류: '$p' (g:gather, 예: 16:0)"
    g=${p%%:*}
    [[ -f $SPLIT_PREFIX-$(( N_FF / g )).gguf ]] ||
        die "split 없음: $SPLIT_PREFIX-$(( N_FF / g )).gguf (python3 regroup_model_split.py 로 생성)"
done

max_np=$(tr ' ' '\n' <<<"$NPS" | sort -n | tail -1)
# 프롬프트 토큰 수는 글자 수 / 3 으로 넉넉히 어림한다 (정확한 검사는 llama-batched 가 시작 시 한다).
prompt_tok=$(( ${#PROMPT} / 3 + 2 ))
need=$(( prompt_tok + (N - prompt_tok) * max_np ))
(( need <= CTX )) || die "CTX=$CTX 가 np=$max_np 에 부족 (필요 약 $need)"

mkdir -p "$OUT_DIR"
{
    git log --oneline -1
    git diff --quiet -- src ggml examples || echo "(커밋 안 된 변경 있음)"
    echo "NPS='$NPS' PAIRS='$PAIRS' ROUNDS=$ROUNDS N=$N CTX=$CTX VB=$VB THREADS=$THREADS"
    echo "PROMPT=$PROMPT"
} | tee "$OUT_DIR/meta.txt"

pswpin() { awk '/^pswpin/ { print $2 }' /proc/vmstat; }

# --- 한 런 -------------------------------------------------------------------
run_one() {
    local np=$1 g=$2 gather=$3 r=$4
    local tag="np${np}_g${g}_gather${gather}_r${r}"
    local log=$OUT_DIR/$tag.log

    if grep -q "decoded .* tokens in" "$log" 2>/dev/null; then
        echo "[skip] $tag"
        return 0
    fi

    # 호스트 RAM 이 빠듯하면 연속 실행이 swap 으로 무너진다. 가능하면 page cache 를 비운다.
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    echo "[run ] $tag  $(date +%T)"
    local sw0
    sw0=$(pswpin)

    # -kvu 필수: 없으면 "sequential split is not supported when there are coupled sequences" 로 실패.
    # batched 는 common_init 을 안 거쳐 워밍업이 없다 (--no-warmup 인자도 이 예제에는 없다).
    env CUDA_VISIBLE_DEVICES=0 \
        KAIROX_PARALLEL=1 KAIROX_GATHER="$gather" \
        KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05 \
    "$BIN" \
        -m "$MODEL" -kairox-ms "$SPLIT_PREFIX-$(( N_FF / g )).gguf" \
        -cffn -fit off -ngl all --no-mmap --no-direct-io \
        -vb "$VB" -t "$THREADS" \
        -c "$CTX" -kvu -np "$np" -n "$N" --ignore-eos -s 42 \
        -p "$PROMPT" \
        >"$log" 2>&1 || echo "  !! $tag 실패 — $log"

    local sw=$(( $(pswpin) - sw0 ))
    echo "pswpin_delta=$sw" >>"$log"
    if (( sw > 1000 )); then
        echo "  !! swap +$sw — 요약에서 제외된다"
    fi

    # batched.cpp 가 -c 를 존중하는지 확인한다. 원본은 n_ctx 를 n_kv_req 로 덮어써 np 마다
    # KV 크기가 달라지고, 그만큼 KAIROX 캐시 용량이 달라져 배치 비교가 무의미해진다.
    local got_ctx
    got_ctx=$(grep -o "llama_context: n_ctx *= *[0-9]*" "$log" | head -1 | grep -o "[0-9]*$")
    if [[ -n $got_ctx && $got_ctx != "$CTX" ]]; then
        die "$tag: n_ctx=$got_ctx != CTX=$CTX — batched.cpp 의 -c 존중 수정이 빌드에 없다 (llama-batched 재빌드)"
    fi
}

# --- 실행 -------------------------------------------------------------------
configs=()
for np in $NPS; do
    for p in $PAIRS; do
        configs+=("$np ${p%%:*} ${p##*:}")   # "np g gather"
    done
done

# 라운드 순환. 라운드마다 조건 순서를 뒤집어 드리프트를 한쪽에 몰지 않는다.
for r in $(seq 1 "$ROUNDS"); do
    if (( r % 2 )); then
        order=("${configs[@]}")
    else
        mapfile -t order < <(printf '%s\n' "${configs[@]}" | tac)
    fi
    for c in "${order[@]}"; do
        run_one $c "$r"
    done
done

# --- 요약 공통: swap 런을 뺀 "tag t/s cached" 목록 -------------------------------
# 로그 줄: main: decoded 1788 tokens in 40.12 s, speed: 44.57 t/s   (전 시퀀스 합산)
collect() {
    local f sw tps cached
    for f in "$OUT_DIR"/np*_r*.log; do
        [[ -e $f ]] || continue
        sw=$(awk -F= '/^pswpin_delta=/ { print $2 }' "$f")
        (( ${sw:-0} > 1000 )) && continue
        tps=$(grep -o "speed: *[0-9.]*" "$f" | tail -1 | awk '{ print $2 }')
        cached=$(grep -o "cached *[0-9]*" "$f" | awk '{ s += $2 } END { print s + 0 }')
        [[ -n $tps ]] && echo "$(basename "$f" .log) $tps $cached"
    done
}

# --- 요약 1: 조건별 처리량 ----------------------------------------------------------
echo
printf "%-4s %-4s %-7s %3s %10s %8s %12s %11s\n" np g gather n "agg t/s" "±95%" "per-seq t/s" "cached(sum)"
collect | awk '
    { split($1, a, "_"); k = a[1] "_" a[2] "_" a[3]
      n[k]++; s[k] += $2; ss[k] += $2 * $2; c[k] = $3
      NP[k] = substr(a[1], 3); G[k] = substr(a[2], 2); GA[k] = substr(a[3], 7) }
    END {
        for (k in n) {
            m  = s[k] / n[k]
            sd = (n[k] > 1) ? sqrt((ss[k] - n[k] * m * m) / (n[k] - 1)) : 0
            printf "%-4s %-4s %-7s %3d %10.2f %8.2f %12.2f %11d\n",
                   NP[k], G[k], (GA[k] ? "on" : "off"), n[k], m, 1.96 * sd / sqrt(n[k]), m / NP[k], c[k]
        }
    }' | sort -k1,1n -k2,2nr -k3,3

# --- 요약 2: np 별 기준 대비 배율 -----------------------------------------------------
#   B1/base   = 전송 병합 기여 (g=16 on  / g=16 off)
#   ours/base = 전체 기여     (g=1  on  / g=16 off)
#   ours/B1   = 입도 기여     (g=1  on  / g=16 on)   <- 배치가 커질수록 오르는지 볼 값
echo
echo "cached(sum) 이 np 사이에 같아야 아래 배율을 np 끼리 비교할 수 있다."
printf "%-5s %10s %10s %10s   %s\n" np "B1/base" "ours/base" "ours/B1" "(t/s: base / B1 / ours)"
collect | awk '
    { split($1, a, "_"); np = substr(a[1], 3); k = substr(a[2], 2) ":" substr(a[3], 7)
      s[np, k] += $2; n[np, k]++; nps[np] = 1 }
    function m(np, k) { return n[np, k] ? s[np, k] / n[np, k] : 0 }
    function r(x, y)  { return (x && y) ? sprintf("%.3fx", x / y) : "-" }
    END {
        for (np in nps) {
            base = m(np, "16:0"); b1 = m(np, "16:1"); ours = m(np, "1:1")
            printf "%-5s %10s %10s %10s   (%.2f / %.2f / %.2f)\n", np, r(b1, base), r(ours, base), r(ours, b1), base, b1, ours
        }
    }' | sort -n