#!/bin/bash
# KAIROX ANB(Adaptive Neuron Balancer) 스윕 드라이버 — dump_activation.sh 래퍼.
#
# 논문 Algorithm 1 Phase 1 이 캐시 거동에 무엇을 하는지를 "카운트 지표"로 잰다.
# 처리량(decode t/s)이 아니라 hit/activation/resident/loads 를 보는 이유는,
# 처리량 노이즈가 +-45% 라 20~30% 미만의 효과는 측정 자체가 안 되기 때문이다.
# 카운트 지표는 같은 프롬프트 + --ignore-eos 면 재현된다.
#
#   bench_models.sh      -> test_kairox.sh       (throughput)
#   bench_activation.sh  -> dump_activation.sh   (activation/hit/resident)
#   bench_group_sweep.sh -> group_sweep.sh       (group_size 스윕)
#   bench_anb.sh         -> dump_activation.sh   (ANB 설정 스윕)              <- 이 파일
#
# 무엇을 보려는 것인가 (ANB_IMPLEMENTATION.md 7장의 가설 검증):
#   1. resident_count 합은 정책과 무관하게 고정이어야 한다
#      (= n_cached_neurons x 총 토큰 수). 아니면 캐시 항상-포화 가정이 깨진 것이다.
#   2. lambda 를 내리면 교체가 늘어야 하는데 tau=(1-lambda) 가 같이 커져 로드를 막는다.
#      이 상쇄가 실재하면 lambda_min 0.10 은 로드가 적고 적중률이 낮게 나온다.
#   3. lambda_min 을 0.50 으로 올리면 그 상쇄가 풀려 적중률이 회복돼야 한다.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root" || exit 1

usage() {
    cat <<'EOF'
usage: [VAR=값 ...] bash bench_anb.sh [simple|full]

  simple  대표 조합 2개만 (anb_off / anb_min050, 동작 확인용)
  full    전체 매트릭스 (기본, 4조합)

기본 조합 (CONFIGS 로 덮어쓸 수 있다)
  이름          ANB  lambda_init  alpha  lambda_min  lambda_max  의미
  anb_off       0    0.67         0.05   -           -           기존 동작 (스왑 예산 적응)
  anb_frozen    1    0.67         0.05   0.67        0.67        널 컨트롤 (lambda 못 움직임)
  anb_min010    1    0.67         0.05   0.10        0.95        논문 Phase 1, 하한 기본값
  anb_min050    1    0.67         0.05   0.50        0.95        논문 Phase 1, tau <= 0.5 로 제한

스윕 축
  CONFIGS       조합 이름 목록            (기본 위 4개)
  VBS           VRAM budget 목록          (기본 "6", vb < gpu_vram)
  PLATFORM      3080 | 3080ti | 4090 | 3070   (기본 3080)

프롬프트 / 생성
  PROMPT_FILE   프롬프트 집합 파일        (기본 ./prompts.txt)
  BENCH_RUNS    조합당 프롬프트 개수      (기본 5, simple 은 2)
  N             프롬프트당 생성 토큰 수   (기본 512)
  REPEAT        조합당 반복 횟수          (기본 1)

기타
  MODEL_DIR     모델 디렉터리             (기본 $HOME/SPIF-GGUF 또는 /root/SPIF-GGUF)
  MODEL         본 모델 .gguf
  MODEL_SPLIT   model-split .gguf         (기본 ...-split-688.gguf = group_size 16)
  OUT_DIR       결과 디렉터리             (기본 ./anb_logs)
  COOLDOWN      런 사이 대기 초           (기본 0)
  DROP_CACHES   1 이면 런 사이에 page cache 를 비운다 (sudo 필요, 기본 0)
                  호스트 RAM 이 모델보다 빠듯하면 연속 실행이 swap 으로 무너진다.
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

platform=${PLATFORM:-3080}
case "$platform" in
3080)   gpu_vram=10; threads=12 ;;
3080ti) gpu_vram=12; threads=12 ;;
4090)   gpu_vram=24; threads=16 ;;
3070)   gpu_vram=8;  threads=7  ;;
*) die "unknown platform '$platform'" ;;
esac
export PLATFORM=$platform

# 조합 정의. "이름:ANB:LAMBDA_INIT:LAMBDA_ADAPT:LAMBDA_MIN:LAMBDA_MAX"
# ANB=0 인 조합에서 min/max 는 쓰이지 않지만 자리는 채워둔다.
config_spec() {
    case "$1" in
    anb_off)    echo "0:0.67:0.05:0.10:0.95" ;;
    anb_frozen) echo "1:0.67:0.05:0.67:0.67" ;;
    anb_min010) echo "1:0.67:0.05:0.10:0.95" ;;
    anb_min050) echo "1:0.67:0.05:0.50:0.95" ;;
    # 참고용 추가 조합
    anb_static) echo "0:0.67:0.00:0.10:0.95" ;; # 적응 전혀 없음
    anb_paper)  echo "1:0.50:0.05:0.50:0.95" ;; # 논문 예시 초기값 0.5
    *) return 1 ;;
    esac
}

if [[ "$run_mode" == "simple" ]]; then
    configs=${CONFIGS:-"anb_off anb_min050"}
    bench_runs=${BENCH_RUNS:-2}
    bench_group=anb_simple
else
    configs=${CONFIGS:-"anb_off anb_frozen anb_min010 anb_min050"}
    bench_runs=${BENCH_RUNS:-5}
    bench_group=anb_sweep
fi

vbs=${VBS:-"6"}

if [[ -n "${MODEL_DIR:-}" ]]; then
    model_dir=$MODEL_DIR
elif [[ -d "$HOME/SPIF-GGUF" ]]; then
    model_dir=$HOME/SPIF-GGUF
else
    model_dir=/root/SPIF-GGUF
fi
export MODEL_DIR=$model_dir

model=${MODEL:-$model_dir/prosparse-llama-2-7b-Q8_0.gguf}
model_split=${MODEL_SPLIT:-$model_dir/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf}

max_tokens=${N:-512}
repeat=${REPEAT:-1}
out_dir=${OUT_DIR:-$repo_root/anb_logs}
force=${FORCE:-0}
rebuild=${REBUILD:-0}
cooldown=${COOLDOWN:-0}
drop_caches=${DROP_CACHES:-0}

summary_csv=$out_dir/anb_summary.csv
lambda_csv=$out_dir/anb_lambda_summary.csv

# dump_activation.sh 로 내려보내는 공통 설정
export N=$max_tokens
export BENCH_RUNS=$bench_runs
export IGNORE_EOS=${IGNORE_EOS:-1} # 토큰 수가 달라지면 카운터 비교가 오염된다
export SUMMARY=0                   # 요약은 이 스크립트가 마지막에 한 번에 낸다
export ANB_TRACE=1                 # lambda 궤적은 항상 남긴다 (ANB=0 이면 빈 파일)
[[ -n "${PROMPT_FILE:-}" ]] && export PROMPT_FILE

# -----------------------------------------------------------------------------
# 준비
# -----------------------------------------------------------------------------

bin=$repo_root/build_rel/bin/llama-completion

if ((rebuild)); then
    rm -rf "$repo_root/build_rel"
fi
if [[ ! -x "$bin" ]]; then
    echo "== build =="
    bash compile_kairox.sh rel || die "빌드 실패"
fi
[[ -x "$bin" ]]          || die "$bin 없음"
[[ -f "$model" ]]        || die "model 없음: $model"
[[ -f "$model_split" ]]  || die "model_split 없음: $model_split"

for c in $configs; do
    config_spec "$c" >/dev/null || die "unknown config '$c' (config_spec 참고)"
done

# --bench-prompt-file 경로는 decode < 16 토큰인 런을 필터링하고 다시 뽑는다.
# N 이 작으면 조건을 영원히 못 맞춰 수백 번 재시도하다 죽는다. 미리 막는다.
((max_tokens >= 32)) || die "N=$max_tokens 은 너무 작다 — bench 경로는 decode>=16 을 요구하므로 32 이상을 쓸 것"

mkdir -p "$out_dir" || die "출력 디렉터리 생성 실패: $out_dir"

model_name=$(basename "$model" .gguf)

echo "=============================================================="
echo " bench_anb.sh ($run_mode)"
echo "=============================================================="
echo " platform    : $platform (gpu_vram=${gpu_vram}GiB, threads=$threads)"
echo " model       : $model_name"
echo " split       : $(basename "$model_split")"
echo " configs     : $configs"
echo " vb          : $vbs"
echo " prompt      : $(basename "${PROMPT_FILE:-$repo_root/prompts.txt}") x ${bench_runs}런"
echo " n / repeat  : $max_tokens / $repeat"
echo " out_dir     : $out_dir"

# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

cool_down() {
    ((drop_caches)) && { sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; }
    ((cooldown > 0)) && sleep "$cooldown"
    return 0
}

tag_of() {
    local cfg=$1 vb=$2 rep=$3 t
    t="${bench_group}__${cfg}__${platform}__completion__${model_name}__vb${vb}"
    ((repeat > 1)) && t="${t}__r${rep}"
    echo "$t"
}

run_one() {
    local cfg=$1 vb=$2 rep=$3
    local spec anb li la lmin lmax tag csv log

    spec=$(config_spec "$cfg")
    IFS=: read -r anb li la lmin lmax <<<"$spec"

    tag=$(tag_of "$cfg" "$vb" "$rep")
    csv=$out_dir/${tag}.csv
    log=$out_dir/${tag}.log

    if [[ -s "$csv" && "$force" != "1" ]]; then
        echo "skip  $tag (이미 있음, FORCE=1 로 재측정)"
        return 0
    fi

    echo
    echo "--------------------------------------------------------------"
    echo " $tag"
    echo "  ANB=$anb lambda=$li alpha=$la range=[$lmin, $lmax]"
    echo "--------------------------------------------------------------"

    cool_down

    BACKEND=kairox VB="$vb" MODEL="$model" MODEL_SPLIT="$model_split" \
        ANB="$anb" LAMBDA_INIT="$li" LAMBDA_ADAPT="$la" \
        LAMBDA_MIN="$lmin" LAMBDA_MAX="$lmax" \
        OUT="$csv" LOG="$log" bash dump_activation.sh || {
        echo "warning: 실행 실패 — 건너뜀 ($tag)" >&2
        return 1
    }
}

for cfg in $configs; do
    for vb in $vbs; do
        ((vb < gpu_vram)) || {
            echo "skip vb=$vb — gpu_vram=$gpu_vram 이상" >&2
            continue
        }
        for ((rep = 1; rep <= repeat; rep++)); do
            run_one "$cfg" "$vb" "$rep"
        done
    done
done

# -----------------------------------------------------------------------------
# 요약 1 — 카운트 지표
# -----------------------------------------------------------------------------

echo
echo "=============================================================="
echo " 요약 1 — 캐시 카운트 지표"
echo "=============================================================="
echo "  hit/res  슬롯 활용도  상주 슬롯-토큰 중 실제로 쓰인 비율"
echo "  hit/act  적중률       활성 뉴런 중 GPU 에 있던 비율  <- CPU 연산으로 직결"
echo "  wasted   낭비율       로드됐다가 한 번도 안 쓰이고 퇴출된 비율 (PCIe 만 소모)"
echo
printf "%-12s %4s %4s %10s %10s %10s %14s %14s\n" \
    config vb rep hit/res hit/act wasted loads resident
echo "config,vb,repeat,hit_over_resident,hit_over_activation,wasted_over_total,total_loads,wasted_loads,total_activation,total_resident" >"$summary_csv"

for cfg in $configs; do
    for vb in $vbs; do
        for ((rep = 1; rep <= repeat; rep++)); do
            csv=$out_dir/$(tag_of "$cfg" "$vb" "$rep").csv
            [[ -s "$csv" ]] || continue
            # CSV 컬럼: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
            awk -F, -v c="$cfg" -v vb="$vb" -v rep="$rep" -v out="$summary_csv" '
                NR>1 { a+=$3; r+=$4; h+=$5; t+=$6; w+=$7 }
                END {
                    hr = (r ? h/r*100 : 0); ha = (a ? h/a*100 : 0); ws = (t ? w/t*100 : 0)
                    printf "%-12s %4s %4s %9.2f%% %9.2f%% %9.2f%% %14d %14d\n", c, vb, rep, hr, ha, ws, t, r
                    printf "%s,%s,%s,%.4f,%.4f,%.4f,%d,%d,%d,%d\n", c, vb, rep, hr, ha, ws, t, w, a, r >> out
                }' "$csv"
        done
    done
done

echo
echo "  주의: resident 열은 정책과 무관하게 (n_cached_neurons x 총 토큰 수) 로 고정이어야 한다."
echo "        조합마다 다르면 캐시 항상-포화 가정이 깨진 것이므로 먼저 그것부터 확인할 것."

# -----------------------------------------------------------------------------
# 요약 2 — lambda 궤적
# -----------------------------------------------------------------------------

echo
echo "=============================================================="
echo " 요약 2 — lambda 궤적 (논문 Figure 11 대응)"
echo "=============================================================="
printf "%-12s %4s %10s %10s %10s %10s\n" config vb "lam(L0-15)" "lam(L16-31)" "io_bound%" "reload/step"
echo "config,vb,lambda_shallow,lambda_deep,io_bound_pct,reload_per_step" >"$lambda_csv"

for cfg in $configs; do
    for vb in $vbs; do
        trace=$out_dir/$(tag_of "$cfg" "$vb" 1)_anb.csv
        [[ -s "$trace" ]] || continue
        # 궤적 컬럼: 1=layer 2=step 3=lambda 4=tau_load 5=io_bound 6=reloads
        # 마지막 1/4 구간만 본다 (수렴 후 동작점).
        awk -F, -v c="$cfg" -v vb="$vb" -v out="$lambda_csv" '
            NR>1 { n[$1]++; if ($2+0 > maxstep[$1]) maxstep[$1]=$2+0 }
            NR>1 { rows[NR]=$0 }
            END {
                for (i=2; i<=NR; i++) {
                    split(rows[i], f, ",")
                    l = f[1]+0
                    if (f[2]+0 < maxstep[l]*0.75) continue   # 수렴 전 구간 제외
                    if (l < 16) { ls += f[3]; lsn++ } else { ld += f[3]; ldn++ }
                    io += f[5]; ion++; rl += f[6]; rln++
                }
                printf "%-12s %4s %10.3f %10.3f %9.1f%% %10.1f\n", c, vb,
                       (lsn?ls/lsn:0), (ldn?ld/ldn:0), (ion?io/ion*100:0), (rln?rl/rln:0)
                printf "%s,%s,%.4f,%.4f,%.2f,%.2f\n", c, vb,
                       (lsn?ls/lsn:0), (ldn?ld/ldn:0), (ion?io/ion*100:0), (rln?rl/rln:0) >> out
            }' "$trace"
    done
done

echo
echo "  ANB=0 조합은 lambda 가 고정이라 궤적이 비어 있는 게 정상이다."
echo "  논문 Figure 11 은 깊은 레이어(16/31)의 lambda 가 0.85 부근에서 안정된다고 보고한다."

# -----------------------------------------------------------------------------
# 요약 3 — 레이어별 적중률
# -----------------------------------------------------------------------------

echo
echo "레이어별 hit/act (조합 비교)"
printf "%-6s" layer
for cfg in $configs; do
    for vb in $vbs; do
        [[ -s "$out_dir/$(tag_of "$cfg" "$vb" 1).csv" ]] && printf "%14s" "$cfg"
    done
done
echo

for l in $(seq 0 31); do
    printf "L%-5d" "$l"
    for cfg in $configs; do
        for vb in $vbs; do
            csv=$out_dir/$(tag_of "$cfg" "$vb" 1).csv
            [[ -s "$csv" ]] || continue
            awk -F, -v L="$l" 'NR>1 && $1==L { a+=$3; h+=$5 }
                END { printf "%13.2f%%", (a?h/a*100:0) }' "$csv"
        done
    done
    echo
done

echo
echo "csv     : $out_dir/"
echo "summary : $summary_csv"
echo "lambda  : $lambda_csv"
