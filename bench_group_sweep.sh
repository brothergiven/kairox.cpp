#!/bin/bash
# KAIROX group_size 스윕 벤치 드라이버 (group_sweep.sh 래퍼).
#
#   bench_models.sh      -> test_kairox.sh       (throughput)
#   bench_activation.sh  -> dump_activation.sh   (activation/hit/resident)
#   bench_group_sweep.sh -> group_sweep.sh       (group_size 스윕)          <- 이 파일
#
# group_sweep.sh 는 group_size 축 하나만 훑는다. 이 래퍼는 그 바깥에
# vb(메모리 압박) 축과 backend(lambda 프로파일) 축을 씌워서
# bench_models.sh 처럼 매트릭스로 돌리고, 결과를 한 표로 모은다.
#
# hw 구성은 test_kairox.sh 의 3080 프로파일(gpu_vram=10, threads=12)로 고정한다.
#
# 환경변수를 export 하는 이유: group_sweep.sh 는 dump_activation.sh 를 호출할 때
# MODEL_SPLIT/OUT/IGNORE_EOS 만 명시적으로 넘긴다. 나머지(PLATFORM/VB/BACKEND/N)는
# 프로세스 환경으로 상속되어야 아래까지 전달된다.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

usage() {
    cat <<'EOF'
usage: [VAR=값 ...] bash bench_group_sweep.sh [simple|full]

  simple  group_size 3개 x 대표 조합 1개 (동작 확인용)
  full    전체 매트릭스 (기본)

고정된 hw 구성 (test_kairox.sh 의 3080 프로파일)
  PLATFORM=3080  gpu_vram=10GiB  threads=12

스윕 축 (환경변수로 덮어쓸 수 있다)
  SIZES     group_size 목록   (기본 "2 4 8 16 32 64 128")
  VBS       VRAM budget 목록  (기본 "5 6 7", vb < 10)
  BACKENDS  lambda 프로파일   (기본 "kairox neuralink")

프롬프트 (bench_models.sh 와 같은 prompts.txt 집합을 기본으로 쓴다)
  PROMPT_FILE 프롬프트 집합 파일  (기본 ./prompts.txt)
  BENCH_RUNS  조합당 프롬프트 개수 (기본 5, simple 은 2)

기타
  MODEL_DIR 모델 디렉터리     (기본 $HOME/SPIF-GGUF 또는 /root/SPIF-GGUF)
  MODEL     본 모델 .gguf
  N         프롬프트당 생성 토큰 수 (기본 512)
  OUT_DIR   결과 디렉터리     (기본 ./group_sweep_logs)
  REGROUP   0 이면 없는 split 을 새로 만들지 않고 건너뛴다 (기본 1)
  REBUILD   1 이면 build_rel 지우고 새로 빌드
  FORCE     1 이면 이미 있는 CSV 도 다시 측정

주의: group_size=1 은 group_identity 가 11008^2*4 = 약 462MiB 라 VRAM 을 크게 먹는다.
      기본 SIZES 에서 뺀 이유다. 필요하면 SIZES="1 2 4 ..." 로 직접 넣을 것.
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

export PLATFORM=3080
gpu_vram=10
threads=12

if [[ -n "${MODEL_DIR:-}" ]]; then
    model_dir=$MODEL_DIR
elif [[ -d "$HOME/SPIF-GGUF" ]]; then
    model_dir=$HOME/SPIF-GGUF
else
    model_dir=/root/SPIF-GGUF
fi
export MODEL_DIR=$model_dir

model=${MODEL:-$model_dir/prosparse-llama-2-7b-Q8_0.gguf}
export MODEL=$model
split_prefix=$model_dir/prosparse-llama-2-7b-sparkinfer-model-split
n_neurons=11008 # prosparse-llama-2-7b 의 n_ff (group_sweep.sh 와 같은 값이어야 한다)

if [[ "$run_mode" == "simple" ]]; then
    sizes=${SIZES:-"8 16 32"}
    vbs=${VBS:-"6"}
    backends=${BACKENDS:-"kairox"}
    bench_runs=${BENCH_RUNS:-2}
else
    sizes=${SIZES:-"2 4 8 16 32 64 128"}
    vbs=${VBS:-"5 6 7"}
    backends=${BACKENDS:-"kairox neuralink"}
    bench_runs=${BENCH_RUNS:-5}
fi

out_dir=${OUT_DIR:-$repo_root/group_sweep_logs}
regroup=${REGROUP:-1}
rebuild=${REBUILD:-0}
force=${FORCE:-0}

export N=${N:-512}
export BENCH_RUNS=$bench_runs      # prompts.txt 에서 앞에서부터 이만큼을 돌린다
export IGNORE_EOS=${IGNORE_EOS:-1} # 토큰 수가 런마다 달라지면 카운터 비교가 오염된다
export SUMMARY=0                   # 조합별 요약은 이 스크립트가 마지막에 한 번에 낸다
[[ -n "${PROMPT_FILE:-}" ]] && export PROMPT_FILE

summary_csv=$out_dir/group_sweep_summary.csv

# -----------------------------------------------------------------------------
# 준비: 빌드 / model-split 생성
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

# split 파일은 permutation 만 담고 있어 1.4MB 수준이고 생성도 몇 초면 끝난다.
# 배포본(-688, group_size=16)을 원본으로 삼아 필요한 group_size 를 미리 만들어 둔다.
base_split=${split_prefix}-$((n_neurons / 16)).gguf
[[ -f "$base_split" ]] || die "기준 model-split 없음: $base_split"

available_sizes=""
for gs in $sizes; do
    if ((n_neurons % gs)); then
        echo "skip group_size=$gs — n_ff($n_neurons) 로 나누어떨어지지 않는다" >&2
        continue
    fi
    split=${split_prefix}-$((n_neurons / gs)).gguf
    if [[ ! -f "$split" ]]; then
        if ((regroup)); then
            echo "== regroup: group_size=$gs (n_group=$((n_neurons / gs))) =="
            python3 "$repo_root/regroup_model_split.py" "$base_split" "$gs" "$split" || {
                echo "warning: group_size=$gs regroup 실패 — 건너뜀" >&2
                continue
            }
        else
            echo "skip group_size=$gs — model-split 없음: $split (REGROUP=1 로 생성)" >&2
            continue
        fi
    fi
    available_sizes="$available_sizes $gs"
done

# xargs 로 앞뒤 공백을 정리한다.
available_sizes=$(echo "$available_sizes" | xargs)
[[ -n "$available_sizes" ]] || die "돌릴 group_size 가 하나도 없다"

mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패: $out_dir"
model_name=$(basename "$model" .gguf)

echo "=============================================================="
echo " bench_group_sweep.sh ($run_mode)"
echo "=============================================================="
echo " platform   : 3080 (gpu_vram=${gpu_vram}GiB, threads=$threads)"
echo " model      : $model_name"
echo " group_size : $available_sizes"
echo " vb         : $vbs"
echo " backends   : $backends"
echo " prompt     : ${PROMPT_FILE:-prompts.txt} x ${bench_runs}런"
echo " n          : $N (프롬프트당)"
echo " out_dir    : $out_dir"

# -----------------------------------------------------------------------------
# 실행: (backend x vb) 마다 group_sweep.sh 를 한 번씩 돌린다
# -----------------------------------------------------------------------------

combo_dir() { echo "$out_dir/${1}__vb${2}"; } # $1=backend $2=vb

for backend in $backends; do
    for vb in $vbs; do
        ((vb < gpu_vram)) || {
            echo "skip vb=$vb — gpu_vram=$gpu_vram 이상" >&2
            continue
        }

        dir=$(combo_dir "$backend" "$vb")
        # bench_models.sh 의 로그 이름 규칙:
        #   <benchmark_group>__<backend>__<hardware>__<generation_mode>__<model>
        log=$out_dir/group_sweep__${backend}__3080__completion__${model_name}__vb${vb}.log

        mkdir -p "$dir"
        ((force)) && rm -f "$dir"/gs*.csv

        echo
        echo "=============================================================="
        echo " backend=$backend  vb=$vb  -> $log"
        echo "=============================================================="

        # group_sweep.sh 는 SIZES/OUT_DIR/MODEL_DIR 을 읽고,
        # BACKEND/VB/PLATFORM/N/MODEL 은 환경 상속으로 dump_activation.sh 까지 내려간다.
        BACKEND="$backend" VB="$vb" SIZES="$available_sizes" OUT_DIR="$dir" \
            bash group_sweep.sh >"$log" 2>&1

        # 어떤 group_size 가 실제로 결과를 남겼는지만 짧게 보여준다.
        for gs in $available_sizes; do
            if [[ -s "$dir/gs${gs}.csv" ]]; then
                printf '  gs=%-4s ok\n' "$gs"
            else
                printf '  gs=%-4s FAIL (로그: %s)\n' "$gs" "$log"
            fi
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
printf "%-10s %4s %6s %10s %10s %12s %12s\n" \
    backend vb gs hit/res hit/act wasted/tot loads

echo "backend,vb,group_size,hit_over_resident,hit_over_activation,wasted_over_total,total_loads,wasted_loads" >"$summary_csv"

for backend in $backends; do
    for vb in $vbs; do
        dir=$(combo_dir "$backend" "$vb")
        for gs in $available_sizes; do
            csv=$dir/gs${gs}.csv
            [[ -s "$csv" ]] || continue
            # CSV 컬럼: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
            awk -F, -v b="$backend" -v vb="$vb" -v gs="$gs" -v out="$summary_csv" '
                NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                END {
                    hr = (r ? h/r*100 : 0); ha = (a ? h/a*100 : 0); ws = (t ? w/t*100 : 0)
                    printf "%-10s %4s %6s %9.2f%% %9.2f%% %11.2f%% %12d\n", b, vb, gs, hr, ha, ws, t
                    printf "%s,%s,%s,%.4f,%.4f,%.4f,%d,%d\n", b, vb, gs, hr, ha, ws, t, w >> out
                }' "$csv"
        done
    done
done

# group_size 를 행, (backend,vb) 를 열로 두는 비교표. 논문 그림에 바로 쓰기 좋다.
print_matrix() {
    local metric=$1 title=$2
    echo
    echo "$title"
    printf "%-8s" gs
    for backend in $backends; do
        for vb in $vbs; do
            [[ -d "$(combo_dir "$backend" "$vb")" ]] || continue
            printf "%14s" "${backend}/vb${vb}"
        done
    done
    echo

    for gs in $available_sizes; do
        printf "gs=%-5s" "$gs"
        for backend in $backends; do
            for vb in $vbs; do
                dir=$(combo_dir "$backend" "$vb")
                [[ -d "$dir" ]] || continue
                csv=$dir/gs${gs}.csv
                if [[ -s "$csv" ]]; then
                    awk -F, -v m="$metric" 'NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                        END {
                            v = (m == "hit_act")  ? (a ? h/a*100 : 0) :
                                (m == "hit_res")  ? (r ? h/r*100 : 0) : (t ? w/t*100 : 0)
                            printf "%13.2f%%", v
                        }' "$csv"
                else
                    printf "%14s" "-"
                fi
            done
        done
        echo
    done
}

print_matrix hit_act "hit / activation  (활성 뉴런 중 GPU 에 있던 비율)"
print_matrix hit_res "hit / resident    (상주 뉴런 중 실제 활성화된 비율)"
print_matrix wasted  "wasted / total    (로드했으나 미사용)"

echo
echo "csv     : $out_dir/<backend>__vb<N>/gs<M>.csv"
echo "summary : $summary_csv"
