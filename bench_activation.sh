#!/bin/bash
# KAIROX Activation 계측 벤치 드라이버 (dump_activation.sh 래퍼).
#
# bench_models.sh 가 test_kairox.sh 를 여러 조합으로 돌리는 것과 같은 역할을
# activation 계측 쪽에서 한다.
#
#   bench_models.sh      -> test_kairox.sh       (throughput)
#   bench_activation.sh  -> dump_activation.sh   (activation/hit/resident)   <- 이 파일
#   bench_group_sweep.sh -> group_sweep.sh       (group_size 스윕)
#
# hw 구성은 test_kairox.sh 의 3080 프로파일(gpu_vram=10, threads=12)로 고정한다.
# 다른 플랫폼으로 돌리고 싶으면 dump_activation.sh 를 직접 PLATFORM= 으로 호출할 것.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

usage() {
    cat <<'EOF'
usage: [VAR=값 ...] bash bench_activation.sh [simple|full]

  simple  대표 조합 1개만 (동작 확인용)
  full    전체 매트릭스 (기본)

고정된 hw 구성 (test_kairox.sh 의 3080 프로파일)
  PLATFORM=3080  gpu_vram=10GiB  threads=12

스윕 축 (환경변수로 덮어쓸 수 있다)
  VBS           VRAM budget 목록          (기본 "4 5 6 7 8", vb < 10)
  BACKENDS      lambda 프로파일 목록      (기본 "kairox neuralink")
  GROUP_SIZES   group_size 목록           (기본 "16" = split-688)

기타
  MODEL_DIR     모델 디렉터리             (기본 $HOME/SPIF-GGUF 또는 /root/SPIF-GGUF)
  MODEL         본 모델 .gguf
  N             생성 토큰 수              (기본 512)
  REPEAT        조합당 반복 횟수          (기본 1)
  OUT_DIR       결과 디렉터리             (기본 ./activation_logs)
  REBUILD       1 이면 build_rel 지우고 새로 빌드
  FORCE         1 이면 이미 있는 CSV 도 다시 측정
EOF
    exit 1
}

die() { echo "error: $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------

run_mode=${1:-full}
[[ $# -le 1 ]] || usage
case "$run_mode" in
simple | full) ;;
*) usage ;;
esac

# hw 고정. dump_activation.sh 로 export 되어 내려간다.
export PLATFORM=3080
gpu_vram=10
threads=12

# 모델 디렉터리: 호스트($HOME)와 컨테이너(/root) 양쪽을 지원한다.
if [[ -n "${MODEL_DIR:-}" ]]; then
    model_dir=$MODEL_DIR
elif [[ -d "$HOME/SPIF-GGUF" ]]; then
    model_dir=$HOME/SPIF-GGUF
else
    model_dir=/root/SPIF-GGUF
fi
export MODEL_DIR=$model_dir

model=${MODEL:-$model_dir/prosparse-llama-2-7b-Q8_0.gguf}
split_prefix=$model_dir/prosparse-llama-2-7b-sparkinfer-model-split
n_neurons=11008 # prosparse-llama-2-7b 의 n_ff. group_size -> n_group 환산에 쓴다.

if [[ "$run_mode" == "simple" ]]; then
    vbs=${VBS:-"6"}
    backends=${BACKENDS:-"kairox"}
    group_sizes=${GROUP_SIZES:-"16"}
    bench_group=simple_validation
else
    vbs=${VBS:-"4 5 6 7 8"}
    backends=${BACKENDS:-"kairox neuralink"}
    group_sizes=${GROUP_SIZES:-"16"}
    bench_group=activation_profile
fi

max_tokens=${N:-512}
repeat=${REPEAT:-1}
out_dir=${OUT_DIR:-$repo_root/activation_logs}
force=${FORCE:-0}
rebuild=${REBUILD:-0}

summary_csv=$out_dir/activation_summary.csv

# dump_activation.sh 는 환경변수로 설정을 받는다. 여기서 export 해두면
# 루프 안에서는 조합마다 달라지는 값(BACKEND/VB/...)만 넘기면 된다.
export N=$max_tokens
export IGNORE_EOS=${IGNORE_EOS:-1} # 런마다 토큰 수가 달라지면 카운터 비교가 오염된다
export SUMMARY=0                   # 조합별 요약은 이 스크립트가 마지막에 한 번에 낸다

# -----------------------------------------------------------------------------
# 준비: 빌드 / 모델 확인
# -----------------------------------------------------------------------------

bin=$repo_root/build_rel/bin/llama-completion

if ((rebuild)); then
    rm -rf "$repo_root/build_rel"
fi
if [[ ! -x "$bin" ]]; then
    echo "== build =="
    bash compile_kairox.sh rel || die "빌드 실패"
fi
[[ -x "$bin" ]] || die "$bin 없음"
[[ -f "$model" ]] || die "model 없음: $model"

# group_size 에 대응하는 model-split 이 없으면 regroup 으로 만들어 둔다.
# split 파일은 permutation 만 담고 있어 1.4MB 정도로 작고 생성도 빠르다.
ensure_split() {
    local gs=$1 n_group split
    n_group=$((n_neurons / gs))
    split=${split_prefix}-${n_group}.gguf

    if [[ ! -f "$split" ]]; then
        local base=${split_prefix}-$((n_neurons / 16)).gguf # 배포본(-688)을 원본으로 삼는다
        [[ -f "$base" ]] || return 1
        echo "== regroup: group_size=$gs (n_group=$n_group) ==" >&2
        python3 "$repo_root/regroup_model_split.py" "$base" "$gs" "$split" >&2 || return 1
    fi
    echo "$split"
}

mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패: $out_dir"

model_name=$(basename "$model" .gguf)

echo "=============================================================="
echo " bench_activation.sh ($run_mode)"
echo "=============================================================="
echo " platform    : 3080 (gpu_vram=${gpu_vram}GiB, threads=$threads)"
echo " model       : $model_name"
echo " backends    : $backends"
echo " vb          : $vbs"
echo " group_size  : $group_sizes"
echo " n / repeat  : $max_tokens / $repeat"
echo " out_dir     : $out_dir"

# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

# 로그에서 decode 처리량을 뽑는다.
#   "llama_perf_context_print:        eval time = ... (   12.34 tokens per second)"
# prompt eval 줄은 prefill 이므로 제외한다.
decode_tps() {
    grep -h "eval time" "$1" 2>/dev/null |
        grep -v "prompt eval" |
        tail -1 |
        sed -n 's/.*, *\([0-9.]*\) tokens per second.*/\1/p'
}

run_one() {
    local backend=$1 vb=$2 gs=$3 rep=$4
    local split tag csv log

    split=$(ensure_split "$gs") || {
        echo "skip group_size=$gs — model-split 생성 실패" >&2
        return 1
    }

    # bench_models.sh 의 로그 이름 규칙을 따른다.
    #   <benchmark_group>__<backend>__<hardware>__<generation_mode>__<model>
    # 여기에 activation 계측 고유 축(gs/vb/rep)을 덧붙였다.
    tag="${bench_group}__${backend}__3080__completion__${model_name}__gs${gs}__vb${vb}"
    ((repeat > 1)) && tag="${tag}__r${rep}"
    csv=$out_dir/${tag}.csv
    log=$out_dir/${tag}.log

    if [[ -s "$csv" && "$force" != "1" ]]; then
        echo "skip  $tag (이미 있음, FORCE=1 로 재측정)"
        return 0
    fi

    echo
    echo "--------------------------------------------------------------"
    echo " $tag"
    echo "--------------------------------------------------------------"

    BACKEND="$backend" VB="$vb" MODEL="$model" MODEL_SPLIT="$split" \
        OUT="$csv" LOG="$log" bash dump_activation.sh || {
        echo "warning: 실행 실패 — 건너뜀 ($tag)" >&2
        return 1
    }

    printf '  decode: %s t/s\n' "$(decode_tps "$log")"
}

for backend in $backends; do
    for vb in $vbs; do
        ((vb < gpu_vram)) || {
            echo "skip vb=$vb — gpu_vram=$gpu_vram 이상" >&2
            continue
        }
        for gs in $group_sizes; do
            for ((rep = 1; rep <= repeat; rep++)); do
                run_one "$backend" "$vb" "$gs" "$rep"
            done
        done
    done
done

# -----------------------------------------------------------------------------
# 요약
# -----------------------------------------------------------------------------

echo
echo "=============================================================="
echo " 요약"
echo "=============================================================="
printf "%-10s %4s %5s %4s %10s %10s %12s %12s %10s\n" \
    backend vb gs rep hit/res hit/act wasted/tot loads decode

echo "backend,vb,group_size,repeat,hit_over_resident,hit_over_activation,wasted_over_total,total_loads,wasted_loads,decode_tps" >"$summary_csv"

for backend in $backends; do
    for vb in $vbs; do
        for gs in $group_sizes; do
            for ((rep = 1; rep <= repeat; rep++)); do
                tag="${bench_group}__${backend}__3080__completion__${model_name}__gs${gs}__vb${vb}"
                ((repeat > 1)) && tag="${tag}__r${rep}"
                csv=$out_dir/${tag}.csv
                [[ -s "$csv" ]] || continue
                tps=$(decode_tps "$out_dir/${tag}.log")

                # CSV 컬럼: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
                awk -F, -v b="$backend" -v vb="$vb" -v gs="$gs" -v rep="$rep" -v tps="${tps:-0}" \
                    -v out="$summary_csv" '
                    NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                    END {
                        hr = (r ? h/r*100 : 0); ha = (a ? h/a*100 : 0); ws = (t ? w/t*100 : 0)
                        printf "%-10s %4s %5s %4s %9.2f%% %9.2f%% %11.2f%% %12d %10s\n", b, vb, gs, rep, hr, ha, ws, t, tps
                        printf "%s,%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%s\n", b, vb, gs, rep, hr, ha, ws, t, w, tps >> out
                    }' "$csv"
            done
        done
    done
done

echo
echo "레이어별 hit/act (조합 비교)"
printf "%-6s" layer
for backend in $backends; do
    for vb in $vbs; do
        for gs in $group_sizes; do
            tag="${bench_group}__${backend}__3080__completion__${model_name}__gs${gs}__vb${vb}"
            ((repeat > 1)) && tag="${tag}__r1" # 레이어별 표는 첫 반복분만 본다
            [[ -s "$out_dir/${tag}.csv" ]] && printf "%14s" "${backend:0:4}/vb$vb/gs$gs"
        done
    done
done
echo

for l in $(seq 0 31); do
    printf "L%-5d" "$l"
    for backend in $backends; do
        for vb in $vbs; do
            for gs in $group_sizes; do
                tag="${bench_group}__${backend}__3080__completion__${model_name}__gs${gs}__vb${vb}"
                ((repeat > 1)) && tag="${tag}__r1"
                csv=$out_dir/${tag}.csv
                [[ -s "$csv" ]] || continue
                awk -F, -v L="$l" 'NR>1 && $1==L { a+=$3; h+=$5 }
                    END { printf "%13.2f%%", (a?h/a*100:0) }' "$csv"
            done
        done
    done
    echo
done

echo
echo "csv     : $out_dir/"
echo "summary : $summary_csv"
