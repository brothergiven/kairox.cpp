#!/bin/bash
cd "$(dirname "$0")" || exit 1
M=${M:-/root/SPIF-GGUF}
mkdir -p dumps3 logs3

# 이름|모델|split접두|기본ng|g1ng|B_iso|B/2|B/4
SPECS=(
 "prosparse-7b|prosparse-llama-2-7b|prosparse-llama-2-7b-sparkinfer-model-split|688|11008|0.063|0.032|0.016"
 "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split|1024|16384|0.018|0.009|0.005"
 "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split|896|14336|0.046|0.023|0.012"
 "SparseQwen2|SparseQwen2-7B|SparseQwen2-7B-sparkinfer-model-split|592|18944|0.022|0.011|0.006"
 "opt-30b|opt-30b-Q4_K_M|opt-30b-sparkinfer-model-split|1024|28672|0.079|0.040|0.020"
)

run() {  # $1=셀 $2=이름 $3=모델 $4=split $5=zerocopy $6=B
  local cell=$1 n=$2 mo=$3 sp=$4 zc=$5 b=$6
  local out="dumps3/${n}__${cell}.csv"
  [[ -s "$out" ]] && { echo "skip $n/$cell"; return; }
  echo "===== $n / $cell  B=$b  ($(basename "$sp"))"
  env KAIROX_ZEROCOPY=$zc KAIROX_GATHER=0 KAIROX_ANB=0 \
      KAIROX_SWAP_BUDGET=$b KAIROX_SWAP_BUDGET_MIN=0.001 \
      LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.00 \
      PLATFORM=3080 VB=0 BENCH_RUNS=3 N=512 SUMMARY=0 \
      MODEL="$M/$mo.gguf" MODEL_SPLIT="$M/$sp.gguf" \
      OUT="$out" LOG="logs3/${n}__${cell}.log" \
      bash dump_activation.sh
  [[ -s "$out" ]] && echo ok || echo "!! CSV 없음 — tail -30 logs3/${n}__${cell}.log"
}

for s in "${SPECS[@]}"; do
  IFS='|' read -r nm mdl base g16 g1 b1 b2 b3 <<<"$s"
  [[ -f "$M/$mdl.gguf" ]] || { echo "skip $nm — 모델 없음"; continue; }
  run D0 "$nm" "$mdl" "${base}-${g16}" 0 "$b1"
  run D1 "$nm" "$mdl" "${base}-${g1}"  1 "$b1"
  run E0 "$nm" "$mdl" "${base}-${g16}" 0 "$b2"
  run E1 "$nm" "$mdl" "${base}-${g1}"  1 "$b2"
  run F0 "$nm" "$mdl" "${base}-${g16}" 0 "$b3"
  run F1 "$nm" "$mdl" "${base}-${g1}"  1 "$b3"
done
echo D_DONE
