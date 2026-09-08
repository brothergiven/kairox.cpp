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

TPOT
  평균 TPOT(ms/token) = 1000 / decode mean 은 항상 요약에 들어간다.
  TOKEN_LATENCY 1 이면 (backend, vb, gs) 마다 per-token latency 패스를 한 번 더
                돌려 p50/p90/p99 분포까지 잰다 (조합당 1회 추가 실행).

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
token_latency=${TOKEN_LATENCY:-0}

# 이 변수는 여기서만 쓴다. 그대로 두면 자식 프로세스(group_sweep.sh -> dump_activation.sh)
# 까지 상속돼 본 측정까지 TPOT 모드로 돌아버린다. 0 으로 덮어써서 내려보내고,
# TPOT 패스를 부를 때만 그 호출에 한해 1 을 준다.
export TOKEN_LATENCY=0
regroup=${REGROUP:-1}
rebuild=${REBUILD:-0}
force=${FORCE:-0}

export N=${N:-512}
export BENCH_RUNS=$bench_runs      # prompts.txt 에서 앞에서부터 이만큼을 돌린다
export IGNORE_EOS=${IGNORE_EOS:-1} # 토큰 수가 런마다 달라지면 카운터 비교가 오염된다
export SUMMARY=0                   # 조합별 요약은 이 스크립트가 마지막에 한 번에 낸다
[[ -n "${PROMPT_FILE:-}" ]] && export PROMPT_FILE

summary_csv=$out_dir/group_sweep_summary.csv
tpot_summary_csv=$out_dir/tpot_summary.csv

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
((token_latency)) && echo " tpot       : per-token latency 패스 켜짐 (조합당 1회 추가 실행)"
echo " out_dir    : $out_dir"

# -----------------------------------------------------------------------------
# 실행: (backend x vb) 마다 group_sweep.sh 를 한 번씩 돌린다
# -----------------------------------------------------------------------------

combo_dir() { echo "$out_dir/${1}__vb${2}"; }               # $1=backend $2=vb
split_for_gs() { echo "${split_prefix}-$((n_neurons / $1)).gguf"; } # $1=group_size

# 조합 로그에서 특정 group_size 의 decode 처리량을 뽑는다.
# 로그에는 gs 들이 순서대로 이어져 있다:
#    group_size=8  (n_group=1376)
#    ...
#      decode mean:    12.34 t/s
# 그래서 "지금 어느 gs 블록인지"를 들고 가면서 그 블록의 decode mean 을 집는다.
# 같은 gs 가 여러 번 나오면(재개/FORCE 재측정) 마지막 값이 남는다.
decode_tps_for_gs() { # $1=log $2=group_size
    [[ -s "$1" ]] || return 0
    awk -v want="$2" '
        /group_size=/ {
            cur = $0
            sub(/.*group_size=/, "", cur)
            sub(/[^0-9].*/, "", cur)
        }
        /decode mean/ && cur == want {
            v = $0
            sub(/.*: */, "", v)
            sub(/ *t\/s.*/, "", v)
        }
        END { print v }' "$1"
}

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

        # 이미 CSV 가 있는 group_size 는 빼고 넘긴다. group_sweep.sh 자체에는
        # 건너뛰기 로직이 없으므로 여기서 목록을 줄여주는 방식으로 재개를 구현한다.
        # (중단 후 같은 명령으로 다시 돌리면 남은 것만 이어서 돈다. FORCE=1 이면 전부 다시)
        pending=""
        for gs in $available_sizes; do
            [[ -s "$dir/gs${gs}.csv" ]] || pending="$pending $gs"
        done
        pending=$(echo "$pending" | xargs)

        if [[ -z "$pending" ]]; then
            echo
            echo "skip  backend=$backend vb=$vb — 모든 group_size 완료 (FORCE=1 로 재측정)"
            continue
        fi

        echo
        echo "=============================================================="
        echo " backend=$backend  vb=$vb  -> $log"
        echo " 남은 group_size : $pending"
        echo "=============================================================="

        # group_sweep.sh 는 SIZES/OUT_DIR/MODEL_DIR 을 읽고,
        # BACKEND/VB/PLATFORM/N/MODEL 은 환경 상속으로 dump_activation.sh 까지 내려간다.
        #
        # 출력을 파일로만 보내면 조합 하나(gs 여러 개)가 다 끝날 때까지 콘솔이 조용하다.
        # tee 로 전체를 로그에 남기면서(이어붙이기), 진행을 알 수 있는 줄만 콘솔로 흘린다.
        #   --line-buffered : 파이프 뒤에서도 줄 단위로 바로 내보낸다. 없으면 grep 이
        #                     4KB 쯤 모아서 내보내기 때문에 여전히 뚝뚝 끊겨 보인다.
        BACKEND="$backend" VB="$vb" SIZES="$pending" OUT_DIR="$dir" \
            bash group_sweep.sh 2>&1 |
            tee -a "$log" |
            grep --line-buffered -E \
                'group_size=|bench run attempt|decode mean|wrote activation dump|^(warning|error):'

        # 어떤 group_size 가 실제로 결과를 남겼는지만 짧게 보여준다.
        for gs in $available_sizes; do
            if [[ -s "$dir/gs${gs}.csv" ]]; then
                printf '  gs=%-4s ok\n' "$gs"
            else
                printf '  gs=%-4s FAIL (로그: %s)\n' "$gs" "$log"
            fi
        done

        ((token_latency)) || continue

        # per-token latency 패스. group_sweep.sh 는 이 모드를 모르므로 여기서
        # gs 를 직접 돌면서 러너를 부른다. activation 덤프가 꺼진 채로 돌기 때문에
        # 위에서 만든 gs*.csv 는 건드리지 않는다.
        # TPOT 패스의 출력은 조합 로그와 분리한다. 같은 파일에 섞으면 여기 찍히는
        # "decode mean" 을 decode_tps_for_gs() 가 그 gs 의 값으로 잘못 집는다.
        tpot_log=$out_dir/tpot__${backend}__3080__completion__${model_name}__vb${vb}.log

        for gs in $available_sizes; do
            tpot_csv=$dir/gs${gs}_tpot.csv
            [[ -s "$tpot_csv" && "$force" != "1" ]] && continue

            echo "  TPOT gs=$gs"
            BACKEND="$backend" VB="$vb" MODEL_SPLIT="$(split_for_gs "$gs")" \
                TOKEN_LATENCY=1 TPOT_OUT="$tpot_csv" \
                LOG="$dir/gs${gs}_tpot.log" \
                bash dump_activation.sh >>"$tpot_log" 2>&1 ||
                echo "  warning: TPOT gs=$gs 실패 (로그: $tpot_log)" >&2
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
printf "%-10s %4s %6s %10s %10s %12s %12s %10s %10s\n" \
    backend vb gs hit/res hit/act wasted/tot loads decode tpot_ms

echo "backend,vb,group_size,hit_over_resident,hit_over_activation,wasted_over_total,total_loads,wasted_loads,decode_tps,tpot_ms" >"$summary_csv"

for backend in $backends; do
    for vb in $vbs; do
        dir=$(combo_dir "$backend" "$vb")
        log=$out_dir/group_sweep__${backend}__3080__completion__${model_name}__vb${vb}.log
        for gs in $available_sizes; do
            csv=$dir/gs${gs}.csv
            [[ -s "$csv" ]] || continue
            tps=$(decode_tps_for_gs "$log" "$gs")

            # TPOT(ms/token) 은 decode 처리량의 역수다. 12 t/s 면 토큰당 1000/12 = 83.3 ms.
            # CSV 컬럼: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
            awk -F, -v b="$backend" -v vb="$vb" -v gs="$gs" -v tps="${tps:-0}" \
                -v out="$summary_csv" '
                NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                END {
                    hr = (r ? h/r*100 : 0); ha = (a ? h/a*100 : 0); ws = (t ? w/t*100 : 0)
                    tpot = (tps + 0 > 0) ? 1000 / tps : 0
                    printf "%-10s %4s %6s %9.2f%% %9.2f%% %11.2f%% %12d %10s %10.2f\n", b, vb, gs, hr, ha, ws, t, tps, tpot
                    printf "%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%s,%.3f\n", b, vb, gs, hr, ha, ws, t, w, tps, tpot >> out
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
                    # 삼항 연산자를 여러 줄에 걸쳐 쓰면 mawk 가 파싱하지 못한다
                    # (gawk 는 통과). 컨테이너 기본 awk 가 mawk 라 if/else 로 쓴다.
                    awk -F, -v m="$metric" 'NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                        END {
                            if (m == "hit_act")      { v = (a ? h/a*100 : 0) }
                            else if (m == "hit_res") { v = (r ? h/r*100 : 0) }
                            else                     { v = (t ? w/t*100 : 0) }
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

# TPOT 평균은 activation CSV 가 아니라 조합 로그의 decode mean 에서 나오므로
# print_matrix 를 그대로 쓰지 못하고 따로 그린다.
echo
echo "TPOT 평균      (ms/token = 1000 / decode mean)"
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
            [[ -d "$(combo_dir "$backend" "$vb")" ]] || continue
            log=$out_dir/group_sweep__${backend}__3080__completion__${model_name}__vb${vb}.log
            tps=$(decode_tps_for_gs "$log" "$gs")
            awk -v tps="${tps:-0}" 'BEGIN {
                if (tps + 0 > 0) { printf "%14.2f", 1000 / tps }
                else             { printf "%14s", "-" }
            }'
        done
    done
    echo
done

if ((token_latency)); then
    echo
    echo "TPOT 분포 (ms/token, per-token 측정)"
    printf "%-10s %4s %6s %8s %10s %10s %10s %10s\n" \
        backend vb gs n mean p50 p90 p99

    echo "backend,vb,group_size,n,mean_ms,p50_ms,p90_ms,p99_ms,max_ms" >"$tpot_summary_csv"

    for backend in $backends; do
        for vb in $vbs; do
            dir=$(combo_dir "$backend" "$vb")
            for gs in $available_sizes; do
                tpot_csv=$dir/gs${gs}_tpot.csv
                [[ -s "$tpot_csv" ]] || continue

                # CSV 컬럼: 1=token_index 2=latency_ms
                # nearest-rank 백분위수 (dump_activation.sh 와 같은 정의)
                tail -n +2 "$tpot_csv" | cut -d, -f2 | sort -n |
                    awk -v b="$backend" -v vb="$vb" -v gs="$gs" -v out="$tpot_summary_csv" '
                        function q(p,   i) {
                            i = int(p * n + 0.999999)
                            if (i < 1) { i = 1 }
                            if (i > n) { i = n }
                            return v[i]
                        }
                        { n = NR; v[NR] = $1; sum += $1 }
                        END {
                            if (n == 0) { exit }
                            printf "%-10s %4s %6s %8d %10.2f %10.2f %10.2f %10.2f\n", b, vb, gs, n, sum/n, q(0.50), q(0.90), q(0.99)
                            printf "%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", b, vb, gs, n, sum/n, q(0.50), q(0.90), q(0.99), v[n] >> out
                        }'
            done
        done
    done
fi

echo
echo "csv     : $out_dir/<backend>__vb<N>/gs<M>.csv"
echo "summary : $summary_csv"
((token_latency)) && echo "tpot    : $tpot_summary_csv"
