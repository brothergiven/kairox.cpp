#!/bin/bash
# 여러 모델에서 group_size 를 스윕해 "입도 -> 적중률 / 실행시간" 곡선을 얻는다.
#
# 왜 하는가
#   KAIROX 의 배포 split 파일명을 보면 n_group 이 전부 1024 이하로 맞춰져 있다.
#     prosparse-7b 688(g=16)  prosparse-13b 864(g=16)  Bamboo 896(g=16)
#     SparseQwen2 592(g=16)   opt-6.7b 1024(g=16)      opt-13b 1024(g=20)
#     opt-30b 1024(g=28)      ReluFalcon-40B 1024(g=32) opt-66b 1024(g=36)
#   즉 고정된 것은 group_size 가 아니라 n_group <= 1024 이고, 이는 논문 8장의
#   "high-performance argsort is most efficient on inputs below 1024 elements,
#    we constrain the neuron group count M to this threshold" 그대로다.
#   g=16 은 워크로드 최적값이 아니라 7B급에서 그 제약이 허용하는 가장 고운 값일 뿐이다.
#
#   단일 모델(prosparse-7b) 측정에서는 g 를 16 -> 2 로 낮추면 적중률이 68.39% -> 80.06% 로
#   올랐다(용량 고정). 이것이 모델 전반에서 성립하는지, 그리고 최적점이 g(뉴런 수)에 붙는지
#   n_group(그룹 개수)에 붙는지를 이 스크립트로 확인한다.
#
# 사용법
#   bash gs_multi_model.sh              # 보유한 모델 전부
#   MODELS="prosparse-llama-2-7b-Q8_0 opt-6.7b" bash gs_multi_model.sh
#   SIZES="4 16 64" BENCH_RUNS=1 bash gs_multi_model.sh    # 축 좁혀 빠르게

set -uo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1
die() { echo "error: $*" >&2; exit 1; }

model_root=${MODEL_ROOT:-$HOME/SPIF-GGUF}
[[ -d "$model_root" ]] || model_root=/root/SPIF-GGUF
[[ -d "$model_root" ]] || die "모델 디렉터리 없음 (MODEL_ROOT 로 지정)"

# ---------------------------------------------------------------------------
# 모델 목록 — "본모델 | 배포 split | vb(GiB)"
#
# split 은 배포된 것 아무거나 하나면 된다. 여기서 n_ff 를 읽고, 다른 group_size 의 split 은
# regroup_model_split.py 로 만들어 쓴다(순열은 그대로, 더 잘게 끊어 읽기만 하므로 재클러스터링 불필요).
#
# vb 는 GPU 와 모델 크기에 맞춰 조정해야 한다. 여기 값은 12GiB(3080 Ti) 기준 어림이다.
# 없는 파일은 자동으로 건너뛴다.
# ---------------------------------------------------------------------------
ENTRIES=(
    "prosparse-llama-2-7b-Q8_0 | prosparse-llama-2-7b-sparkinfer-model-split-688.gguf  | 6"
    "prosparse-llama-2-7b      | prosparse-llama-2-7b-sparkinfer-model-split-688.gguf  | 6"
    "Bamboo-base-v0_1          | Bamboo-base-v0_1-sparkinfer-model-split-896.gguf      | 6"
    "SparseQwen2-7B            | SparseQwen2-7B-sparkinfer-model-split-592.gguf        | 6"
    "opt-6.7b                  | opt-6.7b-sparkinfer-model-split-1024.gguf             | 6"
    "prosparse-llama-2-13b     | prosparse-llama-2-13b-sparkinfer-model-split-864.gguf | 8"
    "opt-13b                   | opt-13b-sparkinfer-model-split-1024.gguf              | 8"
    "opt-30b-Q4_K_M            | opt-30b-sparkinfer-model-split-1024.gguf              | 9"
    "opt-30b-Q8_0              | opt-30b-sparkinfer-model-split-1024.gguf              | 16"
    "ReluFalcon-40B-Q8_0       | ReluFalcon-40B-sparkinfer-model-split-1024.gguf       | 16"
    "opt-66b-Q4_K_M            | opt-66b-sparkinfer-model-split-1024.gguf              | 16"
)

sizes_default="2 4 8 16 32 64 128"
sizes=${SIZES:-$sizes_default}
out_root=${OUT_DIR:-$repo_root/gs_multi_logs}
want=${MODELS:-}

# dump_activation.sh 공통 설정
export PLATFORM=${PLATFORM:-3080}
export BENCH_RUNS=${BENCH_RUNS:-2}
export N=${N:-512}
export IGNORE_EOS=1
export SUMMARY=0
# 정책 격리 조건: 논문에 없는 스왑 예산을 끄고(ANB=1 + lambda 고정),
# one-hit wonder 필터는 실측 최적값 근처(~0)에 둔다. 둘 다 입도와 얽혀 곡선을 왜곡한다.
# as-shipped 로 재려면 ANB=0 TAU_LOAD=0.33 을 주면 된다.
export ANB=${ANB:-1}
export LAMBDA_INIT=${LAMBDA_INIT:-0.67}
export LAMBDA_ADAPT=${LAMBDA_ADAPT:-0.05}
export LAMBDA_MIN=${LAMBDA_MIN:-0.67}
export LAMBDA_MAX=${LAMBDA_MAX:-0.67}
export TAU_LOAD=${TAU_LOAD:-0.0001}

[[ -x "$repo_root/build_rel/bin/llama-completion" ]] || die "빌드 먼저: bash compile_kairox.sh rel"
mkdir -p "$out_root" || die "출력 디렉터리 생성 실패"

read_n_ff() {   # $1 = split 경로 -> "n_ff group_size"
    python3 - "$1" <<'PY' 2>/dev/null
import sys
sys.path.insert(0, "gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["ffn_group_size"]
print(len(r.tensors[0].data), int(f.parts[f.data[0]][0]))
PY
}

echo "=============================================================="
echo " gs_multi_model — 여러 모델 group_size 스윕"
echo "=============================================================="
echo " model_root : $model_root"
echo " sizes      : $sizes"
echo " 조건       : ANB=$ANB tau=$TAU_LOAD lambda=[$LAMBDA_MIN,$LAMBDA_MAX]"
echo " 프롬프트   : ${BENCH_RUNS}개 x ${N}토큰"

declare -a DONE_MODELS
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r name split vb <<<"$entry"
    name=$(echo "$name" | xargs); split=$(echo "$split" | xargs); vb=$(echo "$vb" | xargs)

    [[ -z "$want" || " $want " == *" $name "* ]] || continue

    model=$model_root/$name.gguf
    split_path=$model_root/$split
    if [[ ! -f "$model" ]]; then echo "skip $name — 본 모델 없음" >&2; continue; fi
    if [[ ! -f "$split_path" ]]; then echo "skip $name — split 없음: $split" >&2; continue; fi

    read -r n_ff base_gs < <(read_n_ff "$split_path")
    [[ -n "${n_ff:-}" ]] || { echo "skip $name — split 을 읽지 못함" >&2; continue; }

    out_dir=$out_root/$name
    mkdir -p "$out_dir"
    DONE_MODELS+=("$name|$n_ff|$base_gs|$vb")

    echo
    echo "=============================================================="
    echo " $name   n_ff=$n_ff  배포 g=$base_gs (n_group=$((n_ff / base_gs)))  vb=$vb"
    echo "=============================================================="

    # 배포 group_size 도 반드시 포함시킨다 (비교 기준점)
    for gs in $sizes $base_gs; do
        (( n_ff % gs == 0 )) || { echo "skip g=$gs — n_ff($n_ff) 이 나눠떨어지지 않음"; continue; }
        n_group=$((n_ff / gs))
        csv=$out_dir/gs${gs}.csv
        log=$out_dir/gs${gs}.log
        [[ -s "$csv" ]] && { echo "skip g=$gs (이미 있음)"; continue; }

        gsplit=${split_path%-*}-${n_group}.gguf
        if [[ ! -f "$gsplit" ]]; then
            echo "== regroup: g=$gs (n_group=$n_group) =="
            python3 "$repo_root/regroup_model_split.py" "$split_path" "$gs" "$gsplit" ||
                { echo "warning: regroup 실패 — g=$gs 건너뜀" >&2; continue; }
        fi

        echo "----- $name  g=$gs  n_group=$n_group -----"
        VB="$vb" MODEL="$model" MODEL_SPLIT="$gsplit" OUT="$csv" LOG="$log" \
            bash dump_activation.sh || echo "warning: 실행 실패 — $name g=$gs" >&2
    done
done

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
summarize() {   # $1=out_dir $2=gs $3=n_group $4=ntok
    local csv=$1/gs$2.csv log=$1/gs$2.log tps
    [[ -s "$csv" ]] || return 1
    tps=$(grep -h "decode mean" "$log" 2>/dev/null | tail -1 | sed -n 's/.*: *\([0-9.]*\) *t\/s.*/\1/p')
    awk -F, -v g="$2" -v ng="$3" -v tps="${tps:-0}" -v ntok="$4" '
        NR>1 { a+=$3; h+=$5; t+=$6 }
        END {
            hr = a ? h/a*100 : 0
            printf "%5s %8d %8.2f%% %8.2f%% %12d %8.1f\n", g, ng, hr, 100-hr, t, (tps>0 ? ntok/tps : 0)
        }' "$csv"
}

ntok=$((N * BENCH_RUNS))
echo
for md in "${DONE_MODELS[@]}"; do
    IFS='|' read -r name n_ff base_gs vb <<<"$md"
    echo "=============================================================="
    echo " $name  (n_ff=$n_ff, 배포 g=$base_gs, vb=$vb)"
    echo "=============================================================="
    printf "%5s %8s %9s %9s %12s %8s\n" g n_group 적중률 미스율 로드 시간s
    for gs in $(echo "$sizes $base_gs" | tr ' ' '\n' | sort -n -u); do
        (( n_ff % gs == 0 )) || continue
        summarize "$out_root/$name" "$gs" "$((n_ff / gs))" "$ntok"
    done
    echo
done

# 모델 간 비교 — 최적점이 g 에 붙는가 n_group 에 붙는가
echo "=============================================================="
echo " 모델 간 비교 — 적중률 최고점"
echo "=============================================================="
printf "%-26s %8s %8s %10s %10s %10s\n" 모델 n_ff 배포g 최적g 최적n_group 이득p
for md in "${DONE_MODELS[@]}"; do
    IFS='|' read -r name n_ff base_gs vb <<<"$md"
    best_g=""; best_hr=""; base_hr=""
    for gs in $(echo "$sizes $base_gs" | tr ' ' '\n' | sort -n -u); do
        (( n_ff % gs == 0 )) || continue
        csv=$out_root/$name/gs${gs}.csv
        [[ -s "$csv" ]] || continue
        hr=$(awk -F, 'NR>1{a+=$3;h+=$5} END{printf "%.2f", a?h/a*100:0}' "$csv")
        (( gs == base_gs )) && base_hr=$hr
        if [[ -z "$best_hr" ]] || (( $(python3 -c "print(1 if $hr>$best_hr else 0)") )); then
            best_hr=$hr; best_g=$gs
        fi
    done
    [[ -n "$best_g" ]] || continue
    gain=$(python3 -c "print(f'{$best_hr-${base_hr:-$best_hr}:+.2f}')")
    printf "%-26s %8d %8d %10d %10d %10s\n" "$name" "$n_ff" "$base_gs" "$best_g" "$((n_ff / best_g))" "$gain"
done
echo
echo "  최적 g 가 모델 무관하게 일정하면 -> 입도는 뉴런 수로 결정된다"
echo "  최적 n_group 이 일정하면        -> 그룹 개수가 본질이다"
echo "  최적 g 가 배포 g 와 같으면      -> 논문 선택이 워크로드로도 정당화된다"
echo
echo "csv: $out_root/"
