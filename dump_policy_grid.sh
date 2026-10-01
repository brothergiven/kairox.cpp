#!/bin/bash
# 정책 x 구성 격자로 activation 을 덤프한다.
#
#   셀    정책              입도      전송        되먹임
#   A0    배포본            기본 g    naive       예산 되먹임 (alpha=0.05)
#   A1    배포본            g=1       zerocopy    예산 되먹임 (alpha=0.05)
#   B0    논문 ANB          기본 g    naive       lambda 적응 + tau 필터
#   B1    논문 ANB          g=1       zerocopy    lambda 적응 + tau 필터
#   C0    되먹임 없음       기본 g    naive       없음 (alpha=0)
#   C1    되먹임 없음       g=1       zerocopy    없음 (alpha=0)   <- 우리 제안
#
# 기본은 A0 B0 A1 B1 (20 런, 80 분 안팎). C 는 CELLS 로 켠다.
#
# usage:
#   bash dump_policy_grid.sh
#   CELLS="A0 A1" ONLY=opt-6.7b bash dump_policy_grid.sh
#   CELLS="A0 B0 A1 B1 C0 C1" nohup bash dump_policy_grid.sh > grid.log 2>&1 &
#
# 환경변수:
#   M        모델 디렉터리            (기본 /root/SPIF-GGUF)
#   CELLS    돌릴 셀                  (기본 "A0 B0 A1 B1")
#   ONLY     모델 이름 부분일치 필터  (기본 전체)
#   PLAT VB BR N   dump_activation.sh 로 그대로 전달 (기본 3080 / 0 / 3 / 512)
#   DUMPS LOGS     출력 디렉터리      (기본 dumps3 / logs3)
#   FORCE=1  이미 있는 CSV 도 다시 돌린다

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

M=${M:-/root/SPIF-GGUF}
CELLS=${CELLS:-"A0 B0 A1 B1"}
ONLY=${ONLY:-}
PLAT=${PLAT:-3080}
VB=${VB:-0}
BR=${BR:-3}
N=${N:-512}
DUMPS=${DUMPS:-dumps3}
LOGS=${LOGS:-logs3}
FORCE=${FORCE:-0}

# 이름|모델|기본 split(= 배포본 입도)|g=1 split
MODELS=(
  "prosparse-7b|prosparse-llama-2-7b|prosparse-llama-2-7b-sparkinfer-model-split-688|prosparse-llama-2-7b-sparkinfer-model-split-11008"
  "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split-1024|opt-6.7b-sparkinfer-model-split-16384"
  "SparseQwen2|SparseQwen2-7B|SparseQwen2-7B-sparkinfer-model-split-592|SparseQwen2-7B-sparkinfer-model-split-18944"
  "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split-896|Bamboo-base-v0_1-sparkinfer-model-split-14336"
  "opt-30b|opt-30b-Q4_K_M|opt-30b-sparkinfer-model-split-1024|opt-30b-sparkinfer-model-split-28672"
)

# 셀마다 세 축을 전부 명시한다. 하나라도 빼면 앞 런의 export 가 남아 조용히 오염된다.
cell_env() {
  case "$1" in
  A0) echo "KAIROX_ZEROCOPY=0 KAIROX_GATHER=0 KAIROX_ANB=0 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.05" ;;
  A1) echo "KAIROX_ZEROCOPY=1 KAIROX_GATHER=0 KAIROX_ANB=0 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.05" ;;
  B0) echo "KAIROX_ZEROCOPY=0 KAIROX_GATHER=0 KAIROX_ANB=1 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.50 LAMBDA_ADAPT=0.05 KAIROX_ANB_TRACE=1" ;;
  B1) echo "KAIROX_ZEROCOPY=1 KAIROX_GATHER=0 KAIROX_ANB=1 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.50 LAMBDA_ADAPT=0.05 KAIROX_ANB_TRACE=1" ;;
  C0) echo "KAIROX_ZEROCOPY=0 KAIROX_GATHER=0 KAIROX_ANB=0 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.00" ;;
  C1) echo "KAIROX_ZEROCOPY=1 KAIROX_GATHER=0 KAIROX_ANB=0 KAIROX_SWAP_BUDGET=1.0 LAMBDA_INIT=0.67 LAMBDA_ADAPT=0.00" ;;
  *)  echo "" ;;
  esac
}

for c in $CELLS; do
  [[ -n "$(cell_env "$c")" ]] || { echo "모르는 셀: $c (A0 A1 B0 B1 C0 C1)"; exit 1; }
done

# 0 단계: 필요한 파일이 다 있는지 먼저 본다. 하나라도 없으면 시작하지 않는다.
need_s16=0; need_s1=0
for c in $CELLS; do
  case "$c" in *0) need_s16=1 ;; *1) need_s1=1 ;; esac
done

miss=0
for e in "${MODELS[@]}"; do
  IFS='|' read -r n mo s16 s1 <<<"$e"
  [[ -z "$ONLY" || "$n" == *"$ONLY"* ]] || continue
  files=("$mo")
  ((need_s16)) && files+=("$s16")
  ((need_s1))  && files+=("$s1")
  for f in "${files[@]}"; do
    [[ -f "$M/$f.gguf" ]] || { echo "MISSING $M/$f.gguf"; miss=1; }
  done
done
((miss)) && { echo "== 누락 파일 있음. regroup_model_split.py 로 생성 후 재실행"; exit 1; }

mkdir -p "$DUMPS" "$LOGS"
echo "== 셀=[$CELLS] PLAT=$PLAT VB=$VB BR=$BR N=$N -> $DUMPS/"

run() { # $1=셀 $2=이름 $3=모델 $4=split
  local cell=$1 n=$2 mo=$3 sp=$4
  local out="$DUMPS/${n}__${cell}.csv"
  if [[ -s "$out" && "$FORCE" != "1" ]]; then
    echo "skip  $n/$cell (이미 있음)"
    return
  fi
  echo "===== $n / $cell  ($(basename "$sp"))"
  # dump_activation.sh 는 env 를 -i 없이 쓰므로 여기서 준 변수가 그대로 전달된다.
  env $(cell_env "$cell") \
    PLATFORM="$PLAT" VB="$VB" BENCH_RUNS="$BR" N="$N" SUMMARY=0 \
    MODEL="$M/$mo.gguf" MODEL_SPLIT="$M/$sp.gguf" \
    OUT="$out" LOG="$LOGS/${n}__${cell}.log" \
    bash dump_activation.sh
  [[ -s "$out" ]] && echo "ok" || echo "!! CSV 없음 — tail -30 $LOGS/${n}__${cell}.log"
}

# 모델이 바깥 루프다. 비교할 셀들이 시간상 붙어 있어야 드리프트가 덜 섞인다.
for e in "${MODELS[@]}"; do
  IFS='|' read -r n mo s16 s1 <<<"$e"
  [[ -z "$ONLY" || "$n" == *"$ONLY"* ]] || continue
  for c in $CELLS; do
    case "$c" in
    *0) run "$c" "$n" "$mo" "$s16" ;;
    *1) run "$c" "$n" "$mo" "$s1"  ;;
    esac
  done
done

echo "GRID_DONE  —  요약: python3 sum4.py"
