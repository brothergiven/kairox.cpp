#!/bin/bash
# g=1 의 선택 기구 오버헤드를 제거한 상태에서 D0(기본 g) 대 D1(g=1) 비교. 5 모델.
# D1 은 전송 경로별로 나눈다 — zerocopy 와 gather.
#
# 제거하는 것
#   argsort        KAIROX_NOSORT=1       O(n log n) 전체 정렬 -> O(n) 임계값 선택
#   호스트 스캔    KAIROX_GPU_COMPACT=1  마스크 압축을 GPU 에서
# 제거 못 하는 것
#   마스크 커널    손잡이 없음
#   흩어진 접근    입도에 내재
#
# 셀
#   D0   기본 g + naive     (g=16 에서는 호출이 적어 naive 가 제일 빠르다)
#   D1z  g=1   + zerocopy
#   D1g  g=1   + gather
#
# NOSORT 는 |S| 를 K 에 맞추는 제어가 없다. 모델·입도마다 tau 가 다르므로
# cal 이 짧은 런으로 |S|/K 가 1 에 가장 가까운 tau 를 찾아 taus.txt 에 적는다.
# (tau 는 전송 경로와 무관하다. 입도당 한 번만 보정하면 된다.)
#
# usage:
#   bash d_nooverhead.sh cal     # tau 자동 보정 -> taus.txt      (40 런, 짧음)
#   bash d_nooverhead.sh run     # 제거 후  3셀 x 5모델 x 3반복   (45 런)
#   bash d_nooverhead.sh base    # 제거 전  3셀 x 5모델 x 3반복   (45 런)
#
#   ONLY=opt-6.7b bash d_nooverhead.sh run      # 한 모델만
#   REPS=1        bash d_nooverhead.sh run      # 빠르게 훑기

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

M=${M:-/root/SPIF-GGUF}
ONLY=${ONLY:-}
REPS=${REPS:-3}
TAUS=${TAUS:-taus.txt}
BIN=./build_rel/bin/llama-completion

# 이름|모델|split 접두|기본 g 의 n_group|g=1 의 n_group|동일전송 상한 B
MODELS=(
  "prosparse-7b|prosparse-llama-2-7b|prosparse-llama-2-7b-sparkinfer-model-split|688|11008|0.063"
  "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split|1024|16384|0.018"
  "SparseQwen2|SparseQwen2-7B|SparseQwen2-7B-sparkinfer-model-split|592|18944|0.022"
  "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split|896|14336|0.046"
  "opt-30b|opt-30b-Q4_K_M|opt-30b-sparkinfer-model-split|1024|28672|0.079"
)

[[ -x "$BIN" ]] || { echo "MISSING $BIN — bash compile_kairox.sh rel" >&2; exit 1; }

# $1=모델 $2=split $3=zerocopy $4=gather $5=nosort $6=tau $7=B $8=n $9=bench_runs ${10}=profile
run() {
  env CUDA_VISIBLE_DEVICES=0 KAIROX_PARALLEL=1 \
      KAIROX_ZEROCOPY=$3 KAIROX_GATHER=$4 KAIROX_ANB=0 \
      KAIROX_NOSORT=$5 KAIROX_TAU_LOAD=$6 KAIROX_GPU_COMPACT=$5 \
      KAIROX_PROFILE_PLAN=${10} \
      KAIROX_SWAP_BUDGET=$7 KAIROX_SWAP_BUDGET_MIN=0.001 \
      KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.00 \
    $BIN -m "$M/$1.gguf" -kairox-ms "$M/$2.gguf" \
      -cffn -fit off -ngl all --no-mmap --no-direct-io -vb 0 -no-cnv \
      --repeat-penalty 1.1 -t 12 -s 42 -c 1024 -n "$8" --no-warmup --ignore-eos \
      --bench-prompt-file prompts.txt --bench-runs "$9" --bench-warmup 0 --bench-no-print 2>&1
}

want() { [[ -z "$ONLY" || "$1" == *"$ONLY"* ]]; }

check() {  # 모델 파일 확인. 없으면 KAIROX 초기화에서 segfault 난다.
  local ok=0
  for f in "$M/$2.gguf" "$M/$3-$4.gguf" "$M/$3-$5.gguf"; do
    [[ -f "$f" ]] || { echo "   skip $1 — MISSING $f"; ok=1; }
  done
  return $ok
}

med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

case "${1:-cal}" in

cal)
  : > "$TAUS"
  echo "== tau 이분 탐색 — |S|/K 를 1 에 맞춘다 (|S| 는 tau 에 단조 감소)"
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 b <<<"$e"
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for sp in "$g16" "$g1"; do
      lo=0.0005; hi=0.8; best=; bestd=999; bestr=
      for it in 1 2 3 4 5 6 7; do
        tau=$(awk -v a="$lo" -v b="$hi" 'BEGIN{printf "%.5f", sqrt(a*b)}')   # 로그 중점
        out=$(run "$mo" "$base-$sp" 1 0 1 "$tau" "$b" 64 1 1 | grep -oE '\([0-9.]+ 배\)' | tail -1)
        r=${out//[^0-9.]/}
        if [[ -z "$r" ]]; then
          printf '   %-13s %-6s tau=%-8s (측정 실패)\n' "$nm" "$sp" "$tau"; break
        fi
        printf '   %-13s %-6s tau=%-8s |S|/K=%s\n' "$nm" "$sp" "$tau" "$r"
        d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
        awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$tau; bestd=$d; bestr=$r; }
        # 2% 안에 들면 더 쪼갤 이유가 없다
        awk -v a="$bestd" 'BEGIN{exit !(a<0.02)}' && break
        # |S| 가 크면 tau 를 올려야 한다
        if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then lo=$tau; else hi=$tau; fi
      done
      if [[ -n "$best" ]]; then
        echo "$nm $sp $best $bestr" >> "$TAUS"
        flag=""
        awk -v x="$bestr" 'BEGIN{exit !(x<0.9 || x>1.1)}' && flag="   <<< |S|/K 가 1 에서 10% 넘게 벗어남 — 이 셀은 비교 불가"
        echo "   -> $nm $sp  tau=$best  |S|/K=$bestr$flag"
      fi
    done
  done
  echo; echo "== $TAUS  (모델 split tau |S|/K)"; cat "$TAUS"
  ;;

run|base)
  mode=$1
  nosort=1; [[ "$mode" == base ]] && nosort=0
  [[ "$mode" == run && ! -f "$TAUS" ]] && { echo "$TAUS 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  declare -A R
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 b <<<"$e"
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for cell in "D0 $g16 0 0" "D1z $g1 1 0" "D1g $g1 0 1"; do
      set -- $cell
      tau=0
      if ((nosort)); then
        tau=$(awk -v n="$nm" -v s="$2" '$1==n && $2==s {print $3}' "$TAUS")
        rat=$(awk -v n="$nm" -v s="$2" '$1==n && $2==s {print $4}' "$TAUS")
        awk -v x="${rat:-1}" 'BEGIN{exit !(x<0.9 || x>1.1)}' && \
          echo "   경고: $nm/$1 의 |S|/K=$rat — 상주 집합 크기가 달라 비교가 오염된다" >&2
        [[ -z "$tau" ]] && { echo "   skip $nm/$1 — $TAUS 에 tau 없음"; continue; }
      fi
      vals=()
      for ((rep=1; rep<=REPS; rep++)); do
        printf '%-13s %-4s %-7s tau=%-5s rep%s  ' "$nm" "$1" "$mode" "$tau" "$rep"
        v=$(run "$mo" "$base-$2" "$3" "$4" "$nosort" "$tau" "$b" 512 3 0 \
            | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
        echo "${v:-실패}"
        [[ -n "$v" ]] && vals+=("$v")
      done
      ((${#vals[@]})) && R["$nm|$1"]=$(med "${vals[@]}")
    done
  done
  echo
  printf '%-14s %9s %9s %9s %9s %9s\n' "모델($mode)" D0 D1z D1g D1z/D0 D1g/D0
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm _ <<<"$e"
    want "$nm" || continue
    d0=${R["$nm|D0"]:-}; dz=${R["$nm|D1z"]:-}; dg=${R["$nm|D1g"]:-}
    [[ -z "$d0$dz$dg" ]] && continue
    rz=$([[ -n "$d0" && -n "$dz" ]] && awk -v a="$dz" -v b="$d0" 'BEGIN{printf "%.3f", a/b}' || echo "—")
    rg=$([[ -n "$d0" && -n "$dg" ]] && awk -v a="$dg" -v b="$d0" 'BEGIN{printf "%.3f", a/b}' || echo "—")
    printf '%-14s %9s %9s %9s %9s %9s\n' "$nm" "${d0:-—}" "${dz:-—}" "${dg:-—}" "$rz" "$rg"
  done
  ;;

*)
  echo "usage: bash d_nooverhead.sh {cal|run|base}" >&2
  exit 1
  ;;
esac
