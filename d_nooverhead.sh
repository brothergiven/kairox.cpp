#!/bin/bash
# g=1 의 선택 기구 오버헤드를 제거한 상태에서 D0(기본 g) 대 D1(g=1) 비교.
#
# 제거하는 것
#   argsort        KAIROX_NOSORT=1       O(n log n) 전체 정렬 -> O(n) 임계값 선택
#   호스트 스캔    KAIROX_GPU_COMPACT=1  마스크 압축을 GPU 에서
# 제거 못 하는 것
#   마스크 커널    손잡이 없음
#   흩어진 접근    입도에 내재
#
# NOSORT 는 |S| 를 K 에 맞추는 제어가 없다. 그래서 입도마다 tau 를 따로 맞춰야
# 양쪽이 같은 크기의 상주 집합을 갖는다. 그게 cal 단계다.
#
# usage:
#   bash d_nooverhead.sh cal              # tau 보정: |S|/K 가 1.00 에 가까운 값을 찾는다
#   bash d_nooverhead.sh run 0.10 0.15    # 그 tau 로 3반복 (첫 인자 g=16, 둘째 g=1)
#   bash d_nooverhead.sh base             # 비교 기준: 제거 전 (argsort 그대로)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

M=${M:-/root/SPIF-GGUF}
MODEL=${MODEL:-opt-6.7b}
BASE=${BASE:-opt-6.7b-sparkinfer-model-split}
G16=${G16:-1024}
G1=${G1:-16384}
B=${B:-0.018}          # 동일전송 상한 (모델별로 다르다)
BIN=./build_rel/bin/llama-completion

# 모델이 없으면 KAIROX 초기화에서 segfault 난다 (NULL 체크 없음). 여기서 먼저 막는다.
for f in "$M/$MODEL.gguf" "$M/$BASE-$G16.gguf" "$M/$BASE-$G1.gguf"; do
  [[ -f "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done
[[ -x "$BIN" ]] || { echo "MISSING $BIN — bash compile_kairox.sh rel" >&2; exit 1; }

# $1=split $2=zerocopy $3=nosort $4=tau $5=n $6=bench_runs $7=profile
run() {
  env CUDA_VISIBLE_DEVICES=0 KAIROX_PARALLEL=1 \
      KAIROX_ZEROCOPY=$2 KAIROX_GATHER=0 KAIROX_ANB=0 \
      KAIROX_NOSORT=$3 KAIROX_TAU_LOAD=$4 KAIROX_GPU_COMPACT=$3 \
      KAIROX_PROFILE_PLAN=$7 \
      KAIROX_SWAP_BUDGET=$B KAIROX_SWAP_BUDGET_MIN=0.001 \
      KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.00 \
    $BIN -m "$M/$MODEL.gguf" -kairox-ms "$M/$BASE-$1.gguf" \
      -cffn -fit off -ngl all --no-mmap --no-direct-io -vb 0 -no-cnv \
      --repeat-penalty 1.1 -t 12 -s 42 -c 1024 -n "$5" --no-warmup --ignore-eos \
      --bench-prompt-file prompts.txt --bench-runs "$6" --bench-warmup 0 --bench-no-print 2>&1
}

case "${1:-cal}" in
cal)
  echo "== tau 보정. |S| / K 가 1.00 에 가장 가까운 tau 를 고른다."
  for spec in "D0 $G16 0" "D1 $G1 1"; do
    set -- $spec
    for tau in 0.02 0.05 0.10 0.15 0.20 0.30; do
      printf '%-3s tau=%-5s ' "$1" "$tau"
      run "$2" "$3" 1 "$tau" 128 1 1 | grep -E "decode mean|선택 " | tr -s ' \n' ' '; echo
    done
  done
  ;;
run)
  t16=${2:?g=16 tau}; t1=${3:?g=1 tau}
  for spec in "D0 $G16 0 $t16" "D1 $G1 1 $t1"; do
    set -- $spec
    for rep in 1 2 3; do
      printf '%-3s nosort tau=%-5s rep%s  ' "$1" "$4" "$rep"
      run "$2" "$3" 1 "$4" 512 3 0 | grep -oE 'decode mean:[^,]*'
    done
  done
  ;;
base)
  for spec in "D0 $G16 0" "D1 $G1 1"; do
    set -- $spec
    for rep in 1 2 3; do
      printf '%-3s argsort rep%s  ' "$1" "$rep"
      run "$2" "$3" 0 0 512 3 0 | grep -oE 'decode mean:[^,]*'
    done
  done
  ;;
*)
  echo "usage: bash d_nooverhead.sh {cal|run <tau16> <tau1>|base}" >&2
  exit 1
  ;;
esac
