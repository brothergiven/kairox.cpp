#!/bin/bash
# 이 파일 실행 시 bash로 해석해라

# KAIROX Activation/Resident 계측 러너.
#
# bench_models.sh <-> test_kairox.sh 관계와 같다.
#   group_sweep.sh   : 스윕 드라이버 (여러 조합을 돌린다)
#   dump_activation.sh : 한 조합을 실행하는 러너 (이 파일)
#
# 모든 설정은 환경변수로 받는다. 인자를 쓰지 않는 이유는 드라이버에서
#   VAR=값 bash dump_activation.sh
# 형태로 한 줄에 조합을 넘기기 편해서다.

# -u: 정의되지 않은 변수 사용 시 에러
# -o pipefail : 파이프 중간이 실패해도 전체를 실패로
set -uo pipefail

# 스크립트 파일이 있는 디렉터리를 절대경로로 얻음
# 어느 위치에서 실행되더라도 현재 레포의 root를 얻을 수 있도록.
#  ${BASH_SOURCE[0]}  : 지금 실행중인 스크립트의 경로
#  dirname            : 경로에서 파일 명을 떼고 디렉토리만
#  $( ... )           : 명령어의 출력을 값으로 사용
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# A || B : A가 실패했을 때만 B를 실행한다
cd "$repo_root" || exit 1

# 함수 정의. 호출 시 괄호는 쓰지 않는다.
usage() {
  cat <<'EOF' # 'EOF' 가 나올 때까지를 통째로 문자열로 취급해 cat에 넘긴다.
usage: [VAR=값 ...] bash dump_activation.sh

hw / 실행 프로파일 (test_kairox.sh 의 platform 표와 동일하게 유지한다)
  PLATFORM     3080 | 3080ti | 4090 | 3070      (기본 3080)
  VB           VRAM Budget (GiB). 반드시 gpu_vram 보다 작아야 한다
  THREADS      CPU 스레드 수 (기본: platform 기본값)

backend 프로파일 (test_kairox.sh 의 backend 표와 동일)
  BACKEND      kairox | neuralink                (기본 kairox)
                 kairox    -> lambda_init=0.67  adapt=0.05
                 neuralink -> lambda_init=0.00  adapt=0.00
  LAMBDA_INIT  BACKEND 기본값을 덮어쓴다
  LAMBDA_ADAPT BACKEND 기본값을 덮어쓴다

모델 / 생성
  MODEL_DIR    모델 디렉터리                     (기본 $HOME/SPIF-GGUF)
  MODEL        본 모델 .gguf
  MODEL_SPLIT  model-split .gguf (그룹 크기가 여기에 들어있다)
  N            생성 토큰 수                      (기본 512)
  CTX          컨텍스트 크기                     (기본 1024)
  SEED         샘플링 시드                       (기본 42)
  IGNORE_EOS   1 이면 --ignore-eos 로 N 토큰을 강제 생성 (기본 1)
  PROMPT_FILE  프롬프트 파일 경로 (없으면 내장 프롬프트)

출력
  OUT          activation CSV 경로              (기본 ./kairox_activation.csv)
  LOG          실행 로그 경로. 주면 stdout/stderr 를 여기로 보낸다
  SUMMARY      1 이면 실행 후 요약표를 출력      (기본 1)
EOF
  exit 1
}

die() { echo "error: $*" >&2; exit 1; }

# =============================================================================
# HW 프로파일 — test_kairox.sh 의 set_platform_defaults() 와 값이 같아야 한다.
# =============================================================================

platform=${PLATFORM:-3080} # 환경변수가 지정되어있으면 그 값, 없거나 비어있으면 기본 값 3080

case "$platform" in
3080)
  gpu_vram=10
  platform_threads=12
  vb_default=6
  ;;
3080ti)
  gpu_vram=12
  platform_threads=12
  vb_default=6
  ;;
4090)
  gpu_vram=24
  platform_threads=16
  vb_default=12
  ;;
3070)
  gpu_vram=8
  platform_threads=7
  vb_default=6
  ;;
*)
  echo "error: unknown platform '$platform'" >&2 # 표준 출력이 아니라 표준 에러로 내보낸다
  usage
  ;;
esac

vb=${VB:-$vb_default}
threads=${THREADS:-$platform_threads}

# =============================================================================
# backend 프로파일 — test_kairox.sh 의 set_backend_defaults() 와 값이 같아야 한다.
# =============================================================================

backend=${BACKEND:-kairox}

case "$backend" in
kairox)
  lambda_init_default=0.67
  lambda_adapt_default=0.05
  ;;
neuralink)
  lambda_init_default=0.00
  lambda_adapt_default=0.00
  ;;
*)
  echo "error: unknown backend '$backend' (kairox | neuralink)" >&2
  usage
  ;;
esac

lambda_init=${LAMBDA_INIT:-$lambda_init_default}
lambda_adapt=${LAMBDA_ADAPT:-$lambda_adapt_default}

# =============================================================================
# 모델 / 생성 설정
# =============================================================================

model_dir=${MODEL_DIR:-$HOME/SPIF-GGUF}
model=${MODEL:-$model_dir/prosparse-llama-2-7b-Q8_0.gguf}
model_split=${MODEL_SPLIT:-$model_dir/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf}

max_tokens=${N:-512}
ctx_size=${CTX:-1024}
seed=${SEED:-42}
ignore_eos=${IGNORE_EOS:-1}
summary=${SUMMARY:-1}
csv=${OUT:-$repo_root/kairox_activation.csv}
log=${LOG:-}

# -n "$X" : 문자열이 비어있지 않다면 참
# ${PROMPT_FILE:-} 는 PROMPT_FILE이 존재하지 않으면 빈 문자열을 반환함
if [[ -n "${PROMPT_FILE:-}" ]]; then
  # -f 는 파일이 존재하는지 검사
  # A || B 는 A가 실패하면 B를 실행.
  [[ -f "$PROMPT_FILE" ]] || die "PROMPT_FILE 없음: $PROMPT_FILE"
  prompt=$(<"$PROMPT_FILE") # $(<파일) : 파일 내용을 통쨰로 읽어 값으로 사용.
else
  prompt='Implement and compare multiple sorting algorithms in Python, including quicksort, mergesort, heapsort, and insertion sort. For each algorithm, provide clean implementations, analyze time and space complexity, and discuss when it performs best.
  ```python'
fi

# =============================================================================
# 사전 점검
# =============================================================================

bin=$repo_root/build_rel/bin/llama-completion

#   -x : 실행 가능한 파일로 존재하는가
#   -f : 일반 파일로 존재하는가
# 조건이 참(성공)이면 || 뒤가 실행되지 않고, 거짓(실패)이면 die 가 실행된다.
[[ -x "$bin" ]]         || die "$bin 없음 — 'bash compile_kairox.sh rel' 먼저 실행"
[[ -f "$model" ]]       || die "model 없음: $model"
[[ -f "$model_split" ]] || die "model_split 없음: $model_split"

# (( )) 는 숫자 비교 전용. 안에서는 $ 를 생략하고 C 문법을 그대로 쓴다.
# vb 는 GPU 전체 VRAM 보다 작아야 한다 (나머지는 KV 캐시 등에 필요).
((vb < gpu_vram)) || die "vb=$vb 가 platform=$platform 의 gpu_vram=$gpu_vram 이상이다"

# 계측 코드를 고치고 재빌드를 잊는 실수가 잦아 명시적으로 경고한다.
#   for 변수 in 목록; do ... done  : 파이썬의 for x in [...] 과 같다
#   A -nt B : A 가 B 보다 최신인가 (newer than, 수정시각 비교)
#   && 이므로 조건이 참일 때만 echo 가 실행된다
for src in ggml/src/ggml-cuda/ggml-cuda.cu src/llama-kairox.cpp ggml/include/ggml-kairox.hpp; do
    [[ "$repo_root/$src" -nt "$bin" ]] &&
        echo "warning: $src 가 바이너리보다 최신 — 재빌드 필요할 수 있음" >&2
done

# 이전 실행 결과가 남아 있으면 지운다.
# -f 는 파일이 없어도 에러를 내지 않는 옵션 (force). set -u 와 무관하다.
rm -f "$csv"

echo "platform=$platform gpu_vram=${gpu_vram}GiB threads=$threads vb=${vb}GiB" \
     "backend=$backend lambda=$lambda_init/$lambda_adapt n=$max_tokens ignore_eos=$ignore_eos"
#             ^^^^^^^^^^^^
# ${gpu_vram}GiB 처럼 중괄호를 쓰는 이유: $gpu_vramGiB 라고 쓰면
# 셸이 "gpu_vramGiB" 라는 이름의 변수를 찾아버린다. 변수명 경계를 명시하는 것.
echo "model_split=$(basename "$model_split")  csv=$csv"

# =============================================================================
# 실행
# =============================================================================
#
# env 는 "환경변수를 이렇게 세팅한 상태로 이 프로그램을 실행해라" 는 명령어다.
# C 코드의 getenv("KAIROX_PARALLEL") 이 "1" 을 받게 되는 게 여기서 나온다.
# 이 설정은 실행되는 프로그램에만 적용되고 스크립트나 터미널 환경은 바뀌지 않는다.
#
# KAIROX_PARALLEL=1 은 필수다. 계측 코드는 전부 GGML_OP_RELOAD_PLAN 실행 경로 안에
# 있고, 그 노드는 이 플래그가 켜져야만 그래프에 삽입된다. 끄면 CSV 가 전부 0 이 된다.
#
# get_env_bool() 은 정확히 "0" 또는 "1" 만 받는다. true/yes/on 은 경고 없이 off 로 폴백.
#
# --no-warmup : 워밍업 실행분이 카운터에 누적되지 않게 한다.
# --ignore-eos: 런마다 생성 길이가 달라지면 토큰 스텝 수가 달라져 지표 비교가 오염된다.
#               EOS 를 무시해 항상 N 토큰을 생성시킨다.
# CSV 는 kairox_cache_manager 소멸자에서 기록되므로 정상 종료해야 파일이 남는다
# (Ctrl+C 로 끊으면 파일이 안 생긴다).

env_args=(
    CUDA_VISIBLE_DEVICES=0
    KAIROX_PARALLEL=1
    "KAIROX_DFR_LAMBDA_INIT=$lambda_init"
    "KAIROX_DFR_LAMBDA_ADAPT_RATE=$lambda_adapt"
    KAIROX_DUMP_ACTIVATION=1
    "KAIROX_DUMP_ACTIVATION_PATH=$csv"
)

cmd_args=(
    "$bin"
    -m "$model"
    -kairox-ms "$model_split"
    -cffn -fit off -ngl all
    --no-mmap --no-direct-io
    -vb "$vb"
    -no-cnv
    --repeat-penalty 1.1
    --dry-multiplier 0.6
    -t "$threads"
    -s "$seed"
    -c "$ctx_size"
    -n "$max_tokens"
    -p "$prompt"
    --no-warmup
)

((ignore_eos)) && cmd_args+=(--ignore-eos)

# %q 는 "셸에 다시 붙여넣어도 그대로 돌아가는 형태" 로 따옴표를 붙여 출력한다.
printf '%q ' env "${env_args[@]}" "${cmd_args[@]}"
printf '\n'

if [[ -n "$log" ]]; then
    mkdir -p "$(dirname "$log")"
    env "${env_args[@]}" "${cmd_args[@]}" >"$log" 2>&1
else
    env "${env_args[@]}" "${cmd_args[@]}"
fi

# $? = 직전 명령어의 종료 코드 (0 이면 성공).
# 반드시 바로 다음 줄에서 받아야 한다. 중간에 echo 하나만 끼어도
# 그 echo 의 종료 코드로 덮어써진다.
status=$?
((status == 0)) || echo "warning: 비정상 종료 (exit $status)" >&2

# =============================================================================
# 결과 요약
# =============================================================================

# -s : 파일이 존재하고 크기가 0 보다 큰가.
# die 에 넘기는 문자열이 여러 줄인데, 큰따옴표 안에서는 줄바꿈이 그대로 유지된다.
[[ -s "$csv" ]] || die "CSV 없음: $csv
  - 정상 종료했는지 (소멸자 실행 여부)
  - 로그에 'wrote activation dump to' 가 있는지 확인${log:+ ($log)}"

# 드라이버에서 여러 번 부를 때는 요약이 방해되므로 SUMMARY=0 으로 끌 수 있다.
((summary)) || exit 0

echo
echo "=== 전체 ==="

# awk 는 텍스트를 줄 단위로 훑는 별개의 미니 언어다. CSV 처리에 적합하다.
#   -F,    : 필드 구분자를 콤마로 (CSV)
#   $3     : 그 줄의 3번째 필드. 셸 변수가 아니라 awk 의 문법이다
#   NR     : 현재 줄 번호. NR>1 은 헤더 줄을 건너뛴다는 뜻
#   END{}  : 파일을 다 읽은 뒤 마지막에 한 번 실행
#
# awk 코드를 반드시 작은따옴표로 감싸야 한다.
# 큰따옴표로 쓰면 셸이 $3 을 먼저 치환해버려서 awk 는 빈 값을 보게 된다.
#
# CSV 컬럼: 1=layer 2=neuron 3=activation 4=resident 5=hit 6=total_loads 7=wasted_loads
awk -F, 'NR>1 { act+=$3; res+=$4; hit+=$5; tot+=$6; wst+=$7 }
    END {
        if (res > 0) printf "  hit / resident   : %6.2f%%   상주 뉴런 중 실제 활성화된 비율\n", hit/res*100
        if (act > 0) printf "  hit / activation : %6.2f%%   활성 뉴런 중 GPU 에 있던 비율\n", hit/act*100
        if (tot > 0) printf "  wasted / total   : %6.2f%%   (%d / %d) 로드했으나 미사용\n", wst/tot*100, wst, tot
        else         print  "  wasted / total   : n/a — reload 이벤트 0 (KAIROX_PARALLEL 확인)"
    }' "$csv"

echo
echo "=== 레이어별 ==="

# printf 는 C 의 printf 와 같다. echo 와 달리 서식 지정이 되고 줄바꿈을 직접 넣어야 한다.
#   %-5s : 왼쪽 정렬 5칸 문자열,  %8s : 오른쪽 정렬 8칸
printf "  %-5s %8s %8s %10s\n" layer hit/res hit/act wasted

# t[$1] 처럼 대괄호를 쓰면 awk 의 연상 배열(파이썬 dict)이다. 키는 layer 번호.
#
# gpu_only 레이어는 RELOAD_PLAN 노드가 없어 total_loads 가 0 이다.
# awk 는 0 으로 나누면 프로그램이 죽으므로, 나누기 전에 반드시 걸러야 한다.
# 조건 ? A : B 는 C 의 삼항 연산자와 동일하다.
awk -F, 'NR>1 { a[$1]+=$3; r[$1]+=$4; h[$1]+=$5; t[$1]+=$6; w[$1]+=$7 }
    END {
        for (l in t) {
            hr = (r[l] > 0) ? sprintf("%7.2f%%", h[l]/r[l]*100) : "      -"
            ha = (a[l] > 0) ? sprintf("%7.2f%%", h[l]/a[l]*100) : "      -"
            ws = (t[l] > 0) ? sprintf("%9.2f%%", w[l]/t[l]*100) : "        -"
            printf "  %-5d %8s %8s %10s\n", l, hr, ha, ws
        }
    }' "$csv" | sort -n -k1
#            ^^^^^^^^^^^^^^^
# awk 의 for (l in t) 는 순서를 보장하지 않으므로 sort 로 정렬한다.
#   |     : 앞 명령어의 출력을 뒤 명령어의 입력으로 (파이프)
#   -n    : 숫자 정렬. 없으면 사전순이라 10 이 2 보다 앞에 온다
#   -k1   : 첫 번째 필드 기준

echo
echo "csv: $csv"
