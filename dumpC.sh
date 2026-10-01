#!/bin/bash
cd "$(dirname "$0")" || exit 1
M=${M:-/root/SPIF-GGUF}
mkdir -p dumps3 logs3

# 이름|모델|기본 split|g=1 split
MODELS=(
  "prosparse-7b|prosparse-llama-2-7b|prosparse-llama-2-7b-sparkinfer-model-split-688|prosparse-llama-2-7b-sparkinfer-model-split-11008"
  "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split-1024|opt-6.7b-sparkinfer-model-split-16384"
  "SparseQwen2|SparseQwen2-7B|SparseQwen2-7B-sparkinfer-model-split-592|SparseQwen2-7B-sparkinfer-model-split-18944"
  "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split-896|Bamboo-base-v0_1-sparkinfer-model-split-14336"
  "opt-30b|opt-30b-Q4_K_M|opt-30b-sparkinfer-model-split-1024|opt-30b-sparkinfer-model-split-28672"
)

miss=0
for e in "${MODELS[@]}"; do
  IFS='|' read -r n mo s16 s1 <<<"$e"
  for f in "$mo" "$s16" "$s1"; do
    [[ -f "$M/$f.gguf" ]] || { echo "MISSING $M/$f.gguf"; miss=1; }
  done
done
((miss)) && exit 1

run() {  # $1=셀 $2=이름 $3=모델 $4=split $5=zerocopy
  local cell=$1 n=$2 mo=$3 sp=$4 zc=$5
  local out="dumps3/${n}__${cell}.csv"
  [[ -s "$out" ]] && { echo "skip $n/$cell"; return; }
  echo "===== $n / $cell  ($(basename "$sp"))"
  env KAIROX_ZEROCOPY=$zc KAIROX_GATHER=0 KAIROX_ANB=0 KAIROX_SWAP_BUDGET=1.0 \
      LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.00 \
      PLATFORM=3080 VB=0 BENCH_RUNS=3 N=512 SUMMARY=0 \
      MODEL="$M/$mo.gguf" MODEL_SPLIT="$M/$sp.gguf" \
      OUT="$out" LOG="logs3/${n}__${cell}.log" \
      bash dump_activation.sh
  [[ -s "$out" ]] && echo ok || echo "!! CSV 없음 — tail -30 logs3/${n}__${cell}.log"
}

for e in "${MODELS[@]}"; do
  IFS='|' read -r n mo s16 s1 <<<"$e"
  run C0 "$n" "$mo" "$s16" 0
  run C1 "$n" "$mo" "$s1"  1
done
echo C_DONE
