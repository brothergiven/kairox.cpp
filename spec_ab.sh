#!/bin/bash
#
# speculative decoding 에서의 gather A/B.
#
# batch=1 decode 로는 PCIe 를 포화시킬 수 없다(측정 2~3.3 GB/s, 링크 28 GB/s).
# speculative 는 draft 가 제안한 3~5 토큰을 target 이 한 번에 verify 하므로 한 스텝의
# 활성화 합집합이 커지고, 수락된 토큰만큼 decode 가 빨라져 초당 전송량이 오른다.
# 전송이 실제로 병목이 되는 영역에서 gather 와 group_size 가 어떻게 움직이는지 본다.
#
# usage: [PLATFORM=3070] [VB=6] [N=511] [GS=16] [THREADS=4] [REPS=2] bash spec_ab.sh

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

model_dir=${MODEL_DIR:-$HOME/SPIF-GGUF}
model=${MODEL:-$model_dir/prosparse-llama-2-7b-Q8_0.gguf}
draft=${DRAFT:-$model_dir/Llama-160M-Chat-v1-Q8_0.gguf}
out_dir=${OUT_DIR:-$repo_root/dumps/spec_ab}
gs=${GS:-16}
n=${N:-511}
vb=${VB:-6}
threads=${THREADS:-4}
reps=${REPS:-2}
n_neurons=11008

die() { echo "error: $*" >&2; exit 1; }

split=$model_dir/prosparse-llama-2-7b-sparkinfer-model-split-$((n_neurons / gs)).gguf
[[ -f "$model" ]] || die "model 없음: $model"
[[ -f "$draft" ]] || die "draft 없음: $draft"
[[ -f "$split" ]] || die "model-split 없음: $split"
mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패"

prompt='Implement and compare multiple sorting algorithms in Python, including quicksort, mergesort, heapsort, and insertion sort. For each algorithm, provide clean implementations, analyze time and space complexity, and discuss when it performs best.

```python'

for i in $(seq 1 "$reps"); do
  for g in 0 1; do
    tag=gs${gs}_g${g}_$i
    env \
      CUDA_VISIBLE_DEVICES=0 \
      KAIROX_PARALLEL=1 \
      KAIROX_DFR_LAMBDA_INIT=0.67 \
      KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05 \
      KAIROX_DUMP_ACTIVATION=0 \
      KAIROX_GATHER="$g" \
      ./build_rel/bin/llama-speculative \
      -m "$model" -md "$draft" -ngld all \
      -kairox-ms "$split" \
      -cffn -fit off -ngl all --no-mmap --no-direct-io -vb "$vb" \
      --draft-min 3 --draft-max 5 \
      --repeat-penalty 1.15 --dry-multiplier 0.8 \
      -t "$threads" -s 42 -c 1024 -n "$n" -p "$prompt" \
      --no-warmup --ignore-eos > "$out_dir/$tag.log" 2>&1

    printf "%-14s " "$tag"
    grep -E "encoded|decoded|accept" "$out_dir/$tag.log" | tr '\n' ' '
    echo
  done
done

echo
echo "로그: $out_dir/"
