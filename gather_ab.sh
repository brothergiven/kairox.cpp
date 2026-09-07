#!/bin/bash
#
# gather 경로 A/B — README 5-3 의 2단계.
#
# 결정 단위(group_size)는 그대로 두고 전송 경로만 바꿨을 때의 효과를 잰다.
#   baseline : 그룹당 cudaMemcpyAsync 1회, reload_window_size 마다 동기화
#   gather   : 흩어진 그룹을 pinned staging 으로 모아 H2D 1회 + scatter 커널 1회
#
# 1단계(정확성) : 계측을 켜고 짧게 돌려 두 경로의 CSV 와 생성 텍스트가 같은지 확인한다.
#                 reload plan 자체는 동일하므로 전송 방식만 바뀐 것이면 완전히 일치해야 한다.
# 2단계(처리량) : 계측을 끄고 길게 돌려 tok/s 를 비교한다. 계측은 레이어마다 D2H +
#                 동기화를 추가하므로 켜둔 채로는 처리량 비교가 오염된다.
#
# usage: [PLATFORM=3070] [VB=6] [N=511] [N_CHECK=64] [GS=16] [ONLY=1|2] bash gather_ab.sh

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

model_dir=${MODEL_DIR:-$HOME/SPIF-GGUF}
out_dir=${OUT_DIR:-$repo_root/dumps/gather_ab}
gs=${GS:-16}
n=${N:-511}
n_check=${N_CHECK:-64}
only=${ONLY:-0}
n_neurons=11008

die() { echo "error: $*" >&2; exit 1; }

split=$model_dir/prosparse-llama-2-7b-sparkinfer-model-split-$((n_neurons / gs)).gguf
[[ -f "$split" ]] || die "model-split 없음: $split"

mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패: $out_dir"

# 한 조건을 돌리고 로그를 남긴다.  run <태그> <gather> <window> <dump> <토큰수> <csv경로|->
run() {
    local tag=$1 g=$2 w=$3 d=$4 ntok=$5
    local log=$out_dir/$tag.log

    echo
    echo "=============================================================="
    echo " $tag  (gather=$g window=$w dump=$d n=$ntok)"
    echo "=============================================================="

    MODEL_SPLIT="$split" OUT="$out_dir/$tag.csv" N="$ntok" IGNORE_EOS=1 \
        GATHER="$g" RELOAD_WINDOW="$w" DUMP="$d" \
        bash dump_activation.sh > "$log" 2>&1

    local st=$?
    ((st == 0)) || echo "warning: $tag 비정상 종료 (exit $st) — $log 확인" >&2

    # llama.cpp 의 perf 출력에서 decode 처리량을 뽑는다.
    grep -E "eval time" "$log" | tail -2
    return $st
}

# 1단계: 정확성 -----------------------------------------------------------------
if ((only == 0 || only == 1)); then
    echo "### 1단계: 정확성 검증 (계측 ON, n=$n_check)"
    run chk_base   0 4 1 "$n_check"
    run chk_gather 1 4 1 "$n_check"

    echo
    echo "--- CSV 일치 여부 ---"
    if cmp -s "$out_dir/chk_base.csv" "$out_dir/chk_gather.csv"; then
        echo "OK: 계측 CSV 완전 일치"
    else
        echo "MISMATCH: CSV 가 다르다"
        diff <(head -50 "$out_dir/chk_base.csv") <(head -50 "$out_dir/chk_gather.csv") | head -20
    fi

    echo "--- 생성 텍스트 일치 여부 ---"
    # 로그에서 프롬프트 이후 생성 부분만 비교하기는 까다로우므로 로그 전체에서
    # 타이밍/메모리 등 런마다 달라지는 줄을 걷어내고 비교한다.
    strip() { grep -vE "ms per token|tokens per second|eval time|load time|total time|^ggml_cuda|VRAM|MiB|MB" "$1"; }
    if diff -q <(strip "$out_dir/chk_base.log") <(strip "$out_dir/chk_gather.log") > /dev/null; then
        echo "OK: 생성 텍스트 동일"
    else
        echo "MISMATCH: 출력이 다르다"
        diff <(strip "$out_dir/chk_base.log") <(strip "$out_dir/chk_gather.log") | head -30
    fi
fi

# 2단계: 처리량 -----------------------------------------------------------------
if ((only == 0 || only == 2)); then
    echo
    echo "### 2단계: 처리량 비교 (계측 OFF, n=$n)"
    # w4 가 저자 기본값. w16/w64 는 "동기화 주기가 병목"이라는 가설을 gather 없이 검증한다.
    run perf_w4     0 4  0 "$n"
    run perf_w16    0 16 0 "$n"
    run perf_w64    0 64 0 "$n"
    run perf_gather 1 4  0 "$n"

    echo
    echo "=============================================================="
    echo " 처리량 요약 (gs=$gs, n=$n)"
    echo "=============================================================="
    printf "%-14s %14s %14s\n" 조건 "eval(ms)" "tok/s"
    for tag in perf_w4 perf_w16 perf_w64 perf_gather; do
        log=$out_dir/$tag.log
        [[ -s "$log" ]] || continue
        # 형식: "...: eval time = 32211.55 ms / 511 runs ( 63.04 ms per token, 15.86 tokens per second)"
        # prompt eval 줄과 구분하기 위해 "runs" 가 있는 줄만 본다.
        awk -v tag="$tag" '
            /eval time/ && /runs/ {
                sub(/.*eval time = */, ""); ms = $1 + 0
                match($0, /\( *[0-9.]+ ms per token, *[0-9.]+ tokens per second/)
                seg = substr($0, RSTART, RLENGTH); sub(/.*token, */, "", seg); tps = seg + 0
                found = 1
            }
            END { if (found) printf "%-14s %14.1f %14.2f\n", tag, ms, tps
                  else       printf "%-14s %14s %14s\n", tag, "-", "-" }' "$log"
    done
fi

echo
echo "로그/CSV: $out_dir/"
