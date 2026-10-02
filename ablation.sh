#!/bin/bash
# 뉴런 입도 오버헤드의 절제 실험.  opt-6.7b, Bamboo.
#
# 묻는 것: 뉴런 입도의 비용이 어느 커널에 있고, 각 제거 수단이 그중 얼마를 되돌리는가.
# 각 비교가 한 변수만 움직이도록 셀을 짰다.
#
#   Nns   대 Niso    정렬 단독        (NOSORT)
#   Nnsc  대 Nns     호스트 스캔 단독 (GPU_COMPACT)
#   Niso  대 Giso    입도 단독        (같은 전송량, 같은 alpha)
#   N     대 G       입도 단독        (양쪽 상한 없음)
#   G     대 Ga      되먹임 단독      (그룹 입도)
#   N     대 Niso    상한 단독
#   Ga    대 Gorig   예산 제어 수정판 대 저자 배포본 (KAIROX_CLAMP_INT)
#
# 주의: Ga 는 저자 배포본이 아니다. 배포본의 정수 그룹 예산을 캐시 대비 실수 비율로
# 바꾼 수정판이다 (커밋 1aa07022f). 원본은 (int) 절단 때문에 cur<=20 에서 증가가
# 불가능하고 1 이 흡수 상태라, 76 스텝 안에 레이어-스텝당 1 그룹으로 얼어붙는다.
# Gorig 가 그 원본이다 — "우리가 무엇을 이겼는가" 는 이 칸이 분모여야 한다.
#
# d_nooverhead.sh 는 NOSORT 와 GPU_COMPACT 를 한 플래그로 묶어 둬서 둘의 효과가 섞였다.
# 여기서는 독립 인자다.
#
# 배포본(alpha=0.05)은 되먹임이 예산을 움직이므로 고정 상한이 의미가 없다.
# 그래서 기준 축(Ga/Na)은 상한 없는 셀에만 두고, 제거 사다리는 alpha=0 에서만 돌린다.
#
# usage
#   bash ablation.sh quick   # 새 질문 셋만 빠르게, 보정 재사용     (5모델 ~90분)
#   bash ablation.sh calg    # 그룹 입도 tau 만 덧붙인다 (cal 보존)
#   bash ablation.sh cpu     # CPU 팔 직접 측정 (work / join / evsync)
#   bash ablation.sh threads # -t 스윕 — CPU 임계경로 + 입도의 CPU 효과
#   bash ablation.sh all     # 전 패밀리 순서대로, 로그 파일로      (5모델 ~7시간)
#   bash ablation.sh cal     # 수요 측정 + 상한/tau 이분 탐색          (모델당 ~22분)
#   bash ablation.sh ts      # 벽시계 9셀 x 모델 x 3반복, 라운드로빈    (모델당 ~20분)
#   bash ablation.sh plan    # PROFILE_PLAN  9셀 x 모델 x 1런          (모델당 ~6분)
#   bash ablation.sh nsys    # 커널 분해     9셀 x 모델 x (맨+nsys)    (모델당 ~20분)
#   bash ablation.sh slope   # PLAN_DELAY_US 0/200/400 on Niso        (~3분)
#   bash ablation.sh dump    # 품질 지표(낭비율/적중률)               (~20분)
#
#   ONLY="opt-6.7b opt-30b" bash ablation.sh quick   # 수요비 양 극단 두 모델만 (~1시간)
#   TLIST="2 24" TCELLS="Ga Nnsc" bash ablation.sh threads   # 더 줄여서
#   ONLY=opt-6.7b bash ablation.sh ts      # 한 모델만
#   CELL=Nns      bash ablation.sh nsys    # 한 셀만
#   REPS=1        bash ablation.sh ts      # 빠르게 훑기
#
# 먼저 cal, 그다음 ts 와 plan, 마지막에 nsys.  요약은 python3 abl_sum.py.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

M=${M:-/root/SPIF-GGUF}
ONLY=${ONLY:-}
CELL=${CELL:-}
REPS=${REPS:-3}
# churn(전송량)은 런 길이에 따라 달라진다 — 초기 스텝은 캐시가 채워지는 중이라 높고,
# 길어지면 그 전이가 평균에 묻힌다. Bamboo 의 그룹 전송량이 n=64 에서 492.8,
# n=128 에서 454.4 였다 (8.5% 차이). 그래서 cal 은 비교 런과 같은 길이로 재야 한다.
# cal / ts / plan / slope 가 모두 N 을 쓴다. nsys 만 트레이스 크기 때문에 짧게 간다.
N=${N:-512}
NSYS_N=${NSYS_N:-128}
CAL=${CAL:-abl_cal.txt}
BIN=./build_rel/bin/llama-completion

# 이름|모델|split 접두|그룹 n_group|g=1 n_group
# group_size 는 박아두지 않고 g1/g16 으로 계산한다 — 모델마다 다르다:
#   prosparse 16 / opt-6.7b 16 / Bamboo 16 / SparseQwen2 32 / opt-30b 28
# 16 으로 박아뒀다가 SparseQwen2 와 opt-30b 의 전송량을 2배·1.75배 과소평가할 뻔했다.
MODELS=(
  "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split|1024|16384"
  "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split|896|14336"
  "prosparse-7b|prosparse-llama-2-7b|prosparse-llama-2-7b-sparkinfer-model-split|688|11008"
  "SparseQwen2|SparseQwen2-7B|SparseQwen2-7B-sparkinfer-model-split|592|18944"
  "opt-30b|opt-30b-Q4_K_M|opt-30b-sparkinfer-model-split|1024|28672"
)

# 이름 split zc gather batch 예산 nosort compact alpha clampint
CELLS=(
  # 기준선 — 그룹 입도
  "Gorig g16 0 0 0 one   0 0 0.05 1"   # 배포본 — 정수 그룹 예산
  "Ga    g16 0 0 0 one   0 0 0.05 0"   # 구현본 base — 실수 비율 예산 + 되먹임
  "Gnsc  g16 0 0 0 one   1 1 0.00 0"   # 그룹 + 기구 (alpha=0, 제안과 같은 조건)
  "Gansc g16 0 0 0 one   1 1 0.05 0"   # 그룹 + 되먹임 + 기구 — 현재 최강 구성
  # 제안 사다리 — 뉴런 입도
  "N     g1  1 0 0 one   0 0 0.00 0"   # 묶음 전송, 상한 없음
  "Niso  g1  1 0 0 iso1  0 0 0.00 0"   # + 동일전송 상한
  "Nhalf g1  1 0 0 half1 0 0 0.00 0"   # + 더 조임
  "Nns   g1  1 0 0 iso1  1 0 0.00 0"   # + NOSORT
  "Nnsc  g1  1 0 0 iso1  1 1 0.00 0"   # + GPU 압축  <- 제안 전체
)
# Gnsc / Gansc 는 한 번 내렸다가 되살렸다. 측정해 보니 제안의 진짜 경쟁자다 —
# 두 모델 모두에서 최선이 그룹 입도였고 Nnsc 는 3위와 꼴찌였다.
#   opt-6.7b   Gansc 56.61 > Gorig 55.76 > Nnsc 54.50 > Ga 54.44 > Gnsc 49.76
#   opt-30b    Gns 30.70 > Gansc 30.39 > Gnsc 30.18 > Ga 29.97 > Gorig 29.35 > Nnsc 27.96
# 기준선을 Gorig/Ga 로만 두면 약한 상대로만 재게 된다.
#
# 아직 빼둔 칸 (측정 완료, 되살리려면 한 줄씩 넣으면 된다. tau 는 taug 로 자동 해석)
#   "G     g16 0 0 0 one   0 0 0.00 0"   Ga/G = 되먹임의 값 (opt-6.7b 1.294, opt-30b 1.003)
#   "Giso  g16 0 0 0 iso16 0 0 0.00 0"   Niso/Giso = 바이트당 선택 품질, 5모델 중 4 패
#   "Gns   g16 0 0 0 one   1 0 0.00 0"   Gnsc/Gns = 그룹에서 압축 단독
#   "Na    g1  1 0 0 one   0 0 0.05 0"   Na/N = 0.994~1.002, 되먹임은 g=1 에서 죽는다
#   iso16 보정도 cal 에서 빠져 있다 (Giso 를 되살리면 같이 복원할 것)
# Gns / Gnsc 는 Gans / Gansc 의 alpha=0 짝이다. 이게 없으면 "기구를 양쪽에 얹고
# 입도만 본" 비교(Nnsc 대 Gansc)에 alpha 가 섞여 한 변수 비교가 아니게 된다.
# 셋을 가른다:
#   Gnsc  / G      그룹 입도에서 기구가 버는 것
#   Nnsc  / Gnsc   기구를 양쪽에 얹고 입도만
#   Gansc / Gnsc   기구를 얹은 뒤에도 되먹임이 남는가
# 세 번째가 핵심이다 — 되먹임은 그룹 과적재를 사후에 치우는 장치였고,
# NOSORT 가 이미 전송을 깎아놨으면(opt-6.7b 16.7 -> 14.2 짝) 치울 게 줄어든다.
# 1 에 가까우면 되먹임도 기구에 흡수된다는 뜻이고, 그게 "제어기를 없애고
# 사슬을 짧게" 라는 설계 논지를 직접 지지한다.
# Gans / Gansc 는 배포본(그룹 입도 + 되먹임)에 기구만 얹은 칸이다.
# NOSORT 와 압축은 입도와 독립인데 그룹 입도에서 한 번도 재지 않았다.
#   작으면  우리 기여는 "g=1 이 만든 세금을 g=1 이 되걷는 것" 에 그친다
#   크면    입도와 무관한 독립 기여다
# 그룹 입도는 n_groups 가 592~1024 라 bitonic 경로를 쓰고 정렬이 싸다 — 작을 것으로 본다.
# 확인되어 뺀 칸
#   Nb    cudaMemcpyBatchAsync. Niso 대비 0.865 / 0.733 으로 지고 nsys 아래서는
#         batch API 계측 때문에 0.456 까지 무너져 분해도 못 읽는다. 네 경로 중 최하위.
#   Na    g=1 + 되먹임. 5 모델에서 Na/N = 0.994~1.002 — 되먹임은 g=1 에서 죽는다.
#         N 과 같은 칸을 두 번 도는 셈이라 뺐다.
#   Gans  그룹 + 되먹임 + NOSORT. Gns/Gnsc (alpha=0) 가 들어와 기구 분리를 커버한다.
#         제안의 설정이 alpha=0 이므로 그쪽이 본 비교다.

[[ -x "$BIN" ]] || { echo "MISSING $BIN — bash compile_kairox.sh rel" >&2; exit 1; }
mkdir -p prof abl_logs abl_dumps

want()  { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }   # ONLY="opt-6.7b opt-30b" 도 된다
wantc() { [[ -z "$CELL" || " $CELL " == *" $1 "* ]]; }   # CELL="Gans Gansc" 도 된다
med()   { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

check() {  # 모델 파일 확인. 없으면 KAIROX 초기화에서 segfault 난다.
  local ok=0
  for f in "$M/$2.gguf" "$M/$3-$4.gguf" "$M/$3-$5.gguf"; do
    [[ -f "$f" ]] || { echo "   skip $1 — MISSING $f"; ok=1; }
  done
  return $ok
}

cal_get() { awk -v m="$1" -v k="$2" '$1==m && $2==k {print $3}' "$CAL" 2>/dev/null; }

clocks() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
                      --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

# $1=모델 $2=split전체 $3=zc $4=gather $5=batch $6=budget $7=nosort $8=compact $9=alpha
# ${10}=tau ${11}=n ${12}=bench_runs ${13}=profile_plan ${14}=dump ${15}=delay_us
# stdout 으로 런 로그 전체를 흘린다. 앞에 nsys 같은 래퍼를 붙이려면 WRAP 에 담는다.
# 덤프 경로는 DUMP_OUT 으로 받는다 (KAIROX_DUMP_ACTIVATION=1 일 때만 의미가 있다).
run() {
  env CUDA_VISIBLE_DEVICES=0 KAIROX_PARALLEL=1 KAIROX_ANB=0 \
      KAIROX_ZEROCOPY=$3 KAIROX_GATHER=$4 KAIROX_MEMCPY_BATCH=$5 \
      KAIROX_SWAP_BUDGET=$6 KAIROX_SWAP_BUDGET_MIN=0.001 \
      KAIROX_NOSORT=$7 KAIROX_TAU_LOAD=${10} KAIROX_GPU_COMPACT=$8 \
      KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=$9 \
      KAIROX_PROFILE_PLAN=${13} KAIROX_DUMP_ACTIVATION=${14} KAIROX_PLAN_DELAY_US=${15} \
      KAIROX_DUMP_ACTIVATION_PATH="${DUMP_OUT:-kairox_activation.csv}" \
      KAIROX_CLAMP_INT="${CLAMP_INT:-0}" KAIROX_PROFILE_CPU="${PROF_CPU:-0}" \
    ${WRAP:-} $BIN -m "$M/$1.gguf" -kairox-ms "$M/$2.gguf" \
      -cffn -fit off -ngl all --no-mmap --no-direct-io -vb 0 -no-cnv \
      --repeat-penalty 1.1 -t "${THREADS:-12}" -s 42 -c 1024 -n "${11}" --no-warmup --ignore-eos \
      --bench-prompt-file prompts.txt --bench-runs "${12}" --bench-warmup 0 --bench-no-print 2>&1
}

# 같은 모델 안에서 이미 돈 설정과 완전히 같으면 건너뛴다.
# 상한은 min(두 수요) 에 맞추므로 모델마다 한쪽은 B=1.0 이 되어
# Giso == G 또는 Niso == N 이 된다 — 같은 칸을 두 번 돌 이유가 없다.
declare -A SEEN
dup_key() { echo "$C_SPLIT|$C_ZC|$C_GA|$C_BA|$C_B|$C_NS|$C_TAU|$C_CP|$C_AL|$C_CI"; }
is_dup() {
  local k; k=$(dup_key)
  [[ -n "${SEEN[$1|$k]:-}" ]] && { echo "${SEEN[$1|$k]}"; return 0; }
  SEEN["$1|$k"]=$2; return 1
}

# 셀 한 줄을 풀어 run 인자로 쓸 전역을 채운다.
setcell() {
  local nm=$1 mo=$2 base=$3 g16=$4 g1=$5
  read -r C_NAME C_SPL C_ZC C_GA C_BA C_BUD C_NS C_CP C_AL C_CI <<<"$6"
  export CLAMP_INT="${C_CI:-0}"   # run() 이 환경에서 읽는다
  C_SPLIT="$base-$g16"; [[ "$C_SPL" == g1 ]] && C_SPLIT="$base-$g1"
  C_TAU=0
  if ((C_NS)); then
    # tau 는 입도마다 다르다 — g=1 은 tau, 그룹은 taug
    C_TAU=$(cal_get "$nm" "$([[ "$C_SPL" == g1 ]] && echo tau || echo taug)")
    [[ -z "$C_TAU" ]] && return 1
  fi
  if [[ "$C_BUD" == one ]]; then C_B=1.0; else C_B=$(cal_get "$nm" "$C_BUD"); fi
  [[ -z "$C_B" ]] && return 1
  return 0
}

# PROFILE_PLAN 로그에서 뽑는다.
#   실행 N   레이어-스텝당 실제로 실행된 reload 짝 수  -> 전송량의 직접 측정
#   (N 배)   |S|/K
# "배" 는 멀티바이트라 tr -d 가 불안정하다. bash 치환으로 숫자만 남긴다.
pairs_of() { grep -oE '실행[[:space:]]+[0-9.]+' | tail -1 | grep -oE '[0-9.]+$'; }
sel_of()   { local s; s=$(grep -oE '\([0-9.]+ 배\)' | tail -1); echo "${s//[^0-9.]/}"; }

case "${1:-cal}" in

# ---------------------------------------------------------------- cal
# 동일전송 지점을 양쪽에서 잡는다.
#
# 두 입도의 상한 없는 수요를 먼저 재고, 목표 T = min(그룹 수요, 뉴런 수요) 로 잡아
# T 를 넘는 쪽에만 상한을 건다. 한쪽만 조이면 수요가 더 적은 쪽에는 맞출 방법이 없다 —
# opt-6.7b 는 g=1 수요가 그룹의 47% 라 g=1 을 올려서 맞추는 게 불가능하다
# (Bamboo 는 반대 — g=1 수요가 더 커서 뉴런 쪽을 조여야 한다). 두 모델이 반대 체제다.
#
# 수요는 K 로 계산하지 않고 KAIROX_PROFILE_PLAN 의 "실행" (레이어-스텝당 실제로
# 실행된 reload 짝) 에서 읽는다. 덤프 K 와 PROFILE_PLAN K 가 36% 어긋나 있어
# K 기반 B_iso 는 근거가 없었다. 실행량은 직접 측정이라 그 불일치를 우회한다.
#
# tau: NOSORT 는 |S| 를 K 에 맞추는 제어가 없다. |S|/K -> 1 이 되는 tau 를 찾는다.
cal)
  : > "$CAL"
  export CLAMP_INT=0   # 보정은 수정판 기준으로 한다 (Gorig 는 예산 손잡이를 무시한다)

  # 전송량 한 번 측정. $1=모델 $2=split $3=zerocopy $4=group_size $5=B
  probe() {
    local p; p=$(run "$1" "$2" "$3" 0 0 "$5" 0 0 0.00 0 "$N" 1 1 0 0 | pairs_of)
    [[ -z "$p" ]] && return 1
    awk -v x="$p" -v g="$4" 'BEGIN{printf "%.2f", x*g}'
  }

  # 같은 설정을 REPS 번 재고 중위를 돌려준다. churn 측정은 단발로 11% 흔들린다.
  probe_med() {
    local vs=() v
    for ((r=1; r<=REPS; r++)); do v=$(probe "$@") && vs+=("$v"); done
    ((${#vs[@]})) || return 1
    med "${vs[@]}"
  }

  # 목표 전송량에 상한을 맞춘다 (실행량은 B 에 단조 증가).
  # 탐색 중에는 단발로 훑고(싸게), 고른 B 는 REPS 번 반복으로 검증한다.
  # 측정 노이즈가 11% 이므로 탐색 허용오차를 그보다 좁게 잡으면 노이즈에 과적합한다.
  # $1=모델 $2=split $3=zerocopy $4=group_size $5=목표  ->  "B 달성비" 를 echo
  bisect_b() {
    local lo=0.0002 hi=1.0 best= bestd=999 b v r d
    for it in 1 2 3 4 5 6; do
      b=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.6f", sqrt(a*c)}')
      v=$(probe "$1" "$2" "$3" "$4" "$b")
      [[ -z "$v" ]] && { printf '      B=%-9s (측정 실패)\n' "$b" >&2; break; }
      r=$(awk -v x="$v" -v t="$5" 'BEGIN{printf "%.3f", x/t}')
      printf '      B=%-9s 전송 %-9s 달성/목표=%s\n' "$b" "$v" "$r" >&2
      d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
      awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$b; bestd=$d; }
      awk -v a="$bestd" 'BEGIN{exit !(a<0.08)}' && break
      if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then hi=$b; else lo=$b; fi
    done
    [[ -z "$best" ]] && return 1
    v=$(probe_med "$1" "$2" "$3" "$4" "$best") || return 1
    printf '      검증 %s 반복 중위: 전송 %s\n' "$REPS" "$v" >&2
    echo "$best $(awk -v x="$v" -v t="$5" 'BEGIN{printf "%.3f", x/t}')"
  }

  # 상한을 기록한다. 수요가 목표 이하면 상한이 필요 없으므로 1.0 으로 둔다.
  # $1=모델명 $2=키 $3=수요 $4=목표 $5=모델 $6=split $7=zc $8=gs
  put_cap() {
    if awk -v d="$3" -v t="$4" 'BEGIN{exit !(d <= t*1.02)}'; then
      echo "$1 $2 1.0" >> "$CAL"; echo "$1 ${2}rat 1.000" >> "$CAL"
      printf '   %-6s 수요 %s <= 목표 %s — 상한 불필요 (B=1.0)\n' "$2" "$3" "$4"
      return
    fi
    printf '   %-6s 이분 탐색 (목표 %s)\n' "$2" "$4"
    read -r b r <<<"$(bisect_b "$5" "$6" "$7" "$8" "$4")"
    [[ -z "$b" ]] && { echo "   $2 보정 실패"; return; }
    echo "$1 $2 $b" >> "$CAL"; echo "$1 ${2}rat $r" >> "$CAL"
    f=""; awk -v x="$r" 'BEGIN{exit !(x<0.90 || x>1.10)}' && f="   <<< 10% 밖 — 동일전송 비교 불가"
    printf '   -> %s=%s  달성/목표=%s%s\n' "$2" "$b" "$r" "$f"
  }

  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue

    echo "== $nm : 상한 없는 수요 ($REPS 반복 중위, -n $N)"
    d16=$(probe_med "$mo" "$base-$g16" 0 "$gs" 1.0) || { echo "   실패 — PROFILE_PLAN 출력 없음 (KAIROX_PARALLEL 확인)"; continue; }
    d1=$(probe_med "$mo" "$base-$g1" 1 1 1.0)       || { echo "   실패"; continue; }
    T=$(awk -v a="$d16" -v b="$d1" 'BEGIN{printf "%.2f", (a<b)?a:b}')
    H=$(awk -v t="$T" 'BEGIN{printf "%.2f", t/2}')
    printf '   그룹 %s   뉴런 %s   뉴런/그룹 %s  ->  목표 T=%s (더 적은 쪽)\n' \
      "$d16" "$d1" "$(awk -v a="$d1" -v b="$d16" 'BEGIN{printf "%.3f", a/b}')" "$T"
    for kv in "d16 $d16" "d1 $d1" "T $T"; do echo "$nm $kv" >> "$CAL"; done

    echo "== $nm : 동일전송 상한"
    # iso16 (그룹 쪽 상한) 은 쓰는 칸이 없어 뺐다 — G 계열을 격자에서 내렸다.
    # 되살리려면 put_cap "$nm" iso16 "$d16" "$T" "$mo" "$base-$g16" 0 "$gs"
    put_cap "$nm" iso1  "$d1"  "$T" "$mo" "$base-$g1"  1 1
    echo "== $nm : 절반 상한 (뉴런 쪽만)"
    put_cap "$nm" half1 "$d1"  "$H" "$mo" "$base-$g1"  1 1

    echo "== $nm : tau 이분 탐색 (|S|/K -> 1, |S| 는 tau 에 단조 감소)"
    lo=0.0005; hi=0.8; best=; bestd=999; bestv=
    for it in 1 2 3 4 5 6 7; do
      t=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.5f", sqrt(a*c)}')
      r=$(run "$mo" "$base-$g1" 1 0 0 1.0 1 0 0.00 "$t" "$N" 1 1 0 0 | sel_of)
      [[ -z "$r" ]] && { printf '   tau=%-9s (측정 실패)\n' "$t"; break; }
      printf '   tau=%-9s |S|/K=%s\n' "$t" "$r"
      d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
      awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$t; bestd=$d; bestv=$r; }
      awk -v a="$bestd" 'BEGIN{exit !(a<0.02)}' && break
      if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then lo=$t; else hi=$t; fi
    done
    if [[ -n "$best" ]]; then
      echo "$nm tau $best" >> "$CAL"; echo "$nm taurat $bestv" >> "$CAL"
      f=""; awk -v x="$bestv" 'BEGIN{exit !(x<0.9 || x>1.1)}' && f="   <<< 10% 밖 — 상주 집합이 달라 오염"
      echo "   -> tau=$best  |S|/K=$bestv$f"
    fi

    # 그룹 입도 tau (Gans/Gansc 용). 기구가 입도와 독립인지 재려면 그룹 쪽도 보정해야 한다.
    echo "== $nm : taug 이분 탐색 (그룹 입도)"
    lo=0.0005; hi=0.8; best=; bestd=999; bestv=
    for it in 1 2 3 4 5 6 7; do
      q=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.5f", sqrt(a*c)}')
      r=$(run "$mo" "$base-$g16" 0 0 0 1.0 1 0 0.00 "$q" "$N" 1 1 0 0 | sel_of)
      [[ -z "$r" ]] && { printf '   taug=%-9s (측정 실패)\n' "$q"; break; }
      printf '   taug=%-9s |S|/K=%s\n' "$q" "$r"
      d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
      awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$q; bestd=$d; bestv=$r; }
      awk -v a="$bestd" 'BEGIN{exit !(a<0.02)}' && break
      if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then lo=$q; else hi=$q; fi
    done
    if [[ -n "$best" ]]; then
      echo "$nm taug $best" >> "$CAL"; echo "$nm taugrat $bestv" >> "$CAL"
      f=""; awk -v x="$bestv" 'BEGIN{exit !(x<0.9 || x>1.1)}' && f="   <<< 10% 밖 — 오염"
      echo "   -> taug=$best  |S|/K=$bestv$f"
    fi
    echo
  done
  echo "== $CAL"; cat "$CAL"
  ;;

# ---------------------------------------------------------------- ts
# 벽시계. 반복을 바깥에, 셀을 안에 둬서 드리프트가 셀에 치우치지 않게 한다.
# 매 런 앞에 클럭/온도를 찍는다 — 긴 스윕과 단발 런의 21% 수준 차이가 클럭 탓인지 여기서 드러난다.
ts)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  declare -A R
  : > abl_clocks.txt

  # 예열은 넣지 않는다. 첫 런이 유휴 클럭에서 출발하지만 -n 512 x 3회 는 충분히 길어
  # 발사 직후의 전이 구간이 측정에 묻힌다 (rep1 과 rep2 가 같은 값을 줘서 확인됐다).
  # 짧은 런을 쓰는 nsys 패밀리는 사정이 달라 거기에만 예열을 둔다.
  printf '%-10s %-6s %-6s %-9s %-8s %4s  %-9s  %s\n' 모델 셀 예산 B tau rep t/s clocks
  for ((rep=1; rep<=REPS; rep++)); do
    for e in "${MODELS[@]}"; do
      IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
      want "$nm" || continue
      check "$nm" "$mo" "$base" "$g16" "$g1" >/dev/null || continue
      for c in "${CELLS[@]}"; do
        read -r cn _ <<<"$c"
        wantc "$cn" || continue
        setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || { echo "   skip $nm/$cn — $CAL 에 값 없음"; continue; }
        if twin=$(is_dup "$nm" "$cn"); then
          printf '%-10s %-6s %-6s %-9s %-8s %4s  (== %s, 생략)\n' "$nm" "$cn" "$C_BUD" "$C_B" "$C_TAU" "$rep" "$twin"
          [[ -n "${R["$nm|$twin"]:-}" ]] && R["$nm|$cn"]="${R["$nm|$twin"]}"
          continue
        fi
        ck=$(clocks)
        printf '%-10s %-6s %-6s %-9s %-6s %5s  ' "$nm" "$cn" "$C_BUD" "$C_B" "$C_TAU" "$rep"
        v=$(WRAP= run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
                   "$C_TAU" "$N" 3 0 0 0 \
            | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
        printf '%-9s  %s\n' "${v:-실패}" "$ck"
        echo "$nm $cn $rep ${v:-NA} $ck" >> abl_clocks.txt
        [[ -n "$v" ]] && R["$nm|$cn"]="${R["$nm|$cn"]:-} $v"
      done
    done
  done
  echo
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm _ <<<"$e"; want "$nm" || continue
    echo "== $nm   t/s 중위 (분모 Ga = 배포본 그룹)"
    ga=$(med ${R["$nm|Ga"]:-0})
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      v=${R["$nm|$cn"]:-}; [[ -z "$v" ]] && continue
      m=$(med $v)
      printf '   %-6s %8s   /Ga %s\n' "$cn" "$m" \
        "$(awk -v a="$m" -v b="$ga" 'BEGIN{if(b>0)printf "%.3f", a/b; else print "—"}')"
    done
    echo
  done
  ;;

# ---------------------------------------------------------------- plan
# PROFILE_PLAN. 셀마다 실행량(동일전송 확인), 호스트 스캔/적용 비용, |S|/K 를 한 번에 준다.
# nsys 런에는 이 손잡이를 켜지 않는다 — 호스트 타이머가 끼어들기 때문이다.
plan)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      wantc "$cn" || continue
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || continue
      echo "-- $nm / $cn   B=$C_B tau=$C_TAU alpha=$C_AL nosort=$C_NS compact=$C_CP"
      WRAP= run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
               "$C_TAU" "$N" 1 1 0 0 > "abl_logs/plan__${nm}__${cn}.log" 2>&1
      grep -E 'n_groups=|호출|스캔|적용|합계|선택' "abl_logs/plan__${nm}__${cn}.log" | sed 's/^/   /'
    done
  done
  echo; echo "요약: python3 abl_sum.py"
  ;;

# ---------------------------------------------------------------- nsys
# 커널 분해. 묵은 sqlite 재사용이 오늘 한 번 당한 함정이라 셀마다 다른 태그 + 삭제 + 강제 재추출.
nsys)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  command -v nsys >/dev/null || { echo "nsys 없음" >&2; exit 1; }
  : > abl_nsys_wall.txt

  # 이 패밀리의 런은 -n $N x 1회 로 ts 의 1/4 길이다. 발사 직후의 저클럭 구간이
  # 측정에서 차지하는 비중이 그만큼 커지고, 시리즈의 첫 셀이 그 영향을 혼자 받는다.
  # ts 에서는 런이 길어 묻혔지만 (rep1=rep2 로 확인) 여기서는 못 묻는다.
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" >/dev/null || continue
    echo "예열 (버린다): $nm  $(clocks)"
    WRAP= run "$mo" "$base-$g16" 0 0 0 1.0 0 0 0.00 0 256 1 0 0 0 >/dev/null 2>&1
    echo "   -> $(clocks)"
    break
  done
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      wantc "$cn" || continue
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || continue
      tag="abl_${nm}_${cn}"
      rm -f "prof/$tag".nsys-rep "prof/$tag".sqlite "prof/${tag}"_*.csv
      echo "-- nsys $nm / $cn   B=$C_B tau=$C_TAU"
      # 짝지어진 벽시계. nsys 런과 똑같은 인자(-n $N, 1회)로 바로 앞에 한 번 돈다.
      # ts 패밀리는 -n 512 x 3회라 수준이 다르다 — GPU 유휴 비율은 이 짝으로만 계산한다.
      w=$(WRAP= run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
                 "$C_TAU" "$NSYS_N" 1 0 0 0 \
          | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
      echo "$nm $cn ${w:-NA} $(clocks)" >> abl_nsys_wall.txt
      echo "   짝 벽시계 ${w:-실패} t/s"
      WRAP="nsys profile --force-overwrite=true -o prof/$tag --trace=cuda --sample=none --cpuctxsw=none" \
        run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
            "$C_TAU" "$NSYS_N" 1 0 0 0 > "abl_logs/nsys__${nm}__${cn}.log" 2>&1
      [[ -f "prof/$tag.nsys-rep" ]] || { echo "   리포트 생성 실패 — 로그 확인"; continue; }
      nsys stats --force-export=true --format csv -o "prof/$tag" \
        --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
        "prof/$tag.nsys-rep" > "abl_logs/stats__${nm}__${cn}.log" 2>&1
      ls "prof/${tag}"_*.csv 2>/dev/null | sed 's/^/   /'
      # CSV 가 나왔으면 트레이스를 지운다. abl_sum.py 는 CSV 만 읽는다.
      # 5 모델 x 9 셀이면 .nsys-rep 가 수 GB 씩 쌓여 디스크를 채운다 —
      # 밤새 돌리는 런이 디스크 가득으로 죽는 걸 막는다.
      if compgen -G "prof/${tag}_*.csv" >/dev/null; then
        rm -f "prof/$tag.nsys-rep" "prof/$tag.sqlite"
      else
        echo "   CSV 없음 — 트레이스를 남긴다 (prof/$tag.nsys-rep)"
      fi
    done
  done
  echo; echo "요약: python3 abl_sum.py"
  ;;

# ---------------------------------------------------------------- slope
# 호스트 작업이 임계 경로에 있는지 직접 재는 유일한 계측.
# 주입한 지연만큼 토큰 시간이 그대로 늘면(기울기 1) 완전히 노출, 덜 늘면 겹침에 가려진 것.
# GPU_COMPACT 가 0.9ms 를 지워도 처리량이 -1% 였던 이유를 확정한다.
slope)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      [[ "$cn" == "${CELL:-Niso}" ]] || continue
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || continue
      echo "== $nm / $cn   지연 주입 기울기"
      for us in 0 200 400; do
        printf '   %4s us  ' "$us"
        v=$(WRAP= run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
                   "$C_TAU" "$N" 3 0 0 "$us" \
            | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
        echo "${v:-실패} t/s"
      done
    done
  done
  ;;

# ---------------------------------------------------------------- dump
# 품질 지표(낭비율/적중률). 동일전송 확인은 plan 이 더 싸게 해주므로 여기선 품질만 본다.
# 계측 오버헤드가 섞이므로 t/s 는 읽지 않는다.
dump)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      [[ -n "$CELL" ]] && { wantc "$cn" || continue; } || \
        case "$cn" in Ga|G|Niso|Nhalf|Nns) ;; *) continue ;; esac
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || continue
      out="abl_dumps/${nm}__${cn}.csv"
      [[ -f "$out" ]] && { echo "   skip $nm/$cn — 이미 있음"; continue; }
      echo "-- dump $nm / $cn"
      DUMP_OUT="$out" WRAP= \
        run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
            "$C_TAU" "$N" 1 0 1 0 > "abl_logs/dump__${nm}__${cn}.log" 2>&1
      [[ -f "$out" ]] && echo "   -> $out" || echo "   덤프 안 나옴 — 로그 확인"
    done
  done
  ;;

# ---------------------------------------------------------------- show
# 격자가 제대로 펼쳐지는지 모델 없이 확인한다. 돌리기 전에 한 번 보라.
show)
  printf '%-10s %-6s %-26s %-4s %-4s %-4s %-9s %-8s %-7s %-8s %s\n' \
         모델 셀 split zc gath bat B tau nosort compact alpha
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      wantc "$cn" || continue
      if setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c"; then
        printf '%-10s %-6s %-26s %-4s %-4s %-4s %-9s %-8s %-7s %-8s %s\n' \
          "$nm" "$C_NAME" "...-${C_SPLIT##*-}" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_TAU" \
          "$C_NS" "$C_CP" "$C_AL/$C_CI"
      else
        printf '%-10s %-6s  <<< %s 에 값 없음 (cal 먼저)\n' "$nm" "$cn" "$CAL"
      fi
    done
  done
  ;;

# ---------------------------------------------------------------- calg
# 그룹 입도 tau 만 덧붙인다. cal 은 파일을 지우고 다시 쓰므로 (수요 측정 + 상한 탐색)
# 이미 유효한 보정이 있을 때 taug 하나 때문에 전체를 다시 돌릴 이유가 없다.
calg)
  export CLAMP_INT=0
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    [[ -n "$(cal_get "$nm" taug)" ]] && { echo "   skip $nm — taug 이미 있음"; continue; }
    echo "== $nm : taug 이분 탐색 (그룹 입도)"
    lo=0.0005; hi=0.8; best=; bestd=999; bestv=
    for it in 1 2 3 4 5 6 7; do
      q=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.5f", sqrt(a*c)}')
      r=$(run "$mo" "$base-$g16" 0 0 0 1.0 1 0 0.00 "$q" "$N" 1 1 0 0 | sel_of)
      [[ -z "$r" ]] && { printf '   taug=%-9s (측정 실패)\n' "$q"; break; }
      printf '   taug=%-9s |S|/K=%s\n' "$q" "$r"
      d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
      awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$q; bestd=$d; bestv=$r; }
      awk -v a="$bestd" 'BEGIN{exit !(a<0.02)}' && break
      if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then lo=$q; else hi=$q; fi
    done
    if [[ -n "$best" ]]; then
      echo "$nm taug $best" >> "$CAL"; echo "$nm taugrat $bestv" >> "$CAL"
      f=""; awk -v x="$bestv" 'BEGIN{exit !(x<0.9 || x>1.1)}' && f="   <<< 10% 밖 — 오염"
      echo "   -> taug=$best  |S|/K=$bestv$f"
    fi
  done
  ;;

# ---------------------------------------------------------------- quick
# 세 질문을 빠르게. 기존 보정을 재사용하고 새 칸만 돌린다.
#   1. 배포본(그룹+되먹임)에 NOSORT/압축을 얹으면?   Gans / Gansc
#   2. CPU 를 줄이면 판정이 바뀌나?                   threads
#   3. 입도가 CPU 연산시간을 바꾸나?                  threads 의 셀별 기울기
# 기본은 N=256, REPS=1 로 빠르게 훑는다 (노이즈 2.8% 를 이미 알고 있다).
quick)
  log=${ALL_LOG:-abl_quick_$(date +%m%d_%H%M).log}
  export N=${N:-256} REPS=${REPS:-1}
  echo "전체 로그: $log   (N=$N REPS=$REPS)"
  {
    echo "시작 $(date '+%F %T')   모델 ${#MODELS[@]} 개"
    echo; echo "######################## ts (Gans/Gansc)  $(date '+%F %T')"
    CELL="Gorig Ga Gnsc Gansc Niso Nnsc" bash ablation.sh ts
    echo; echo "######################## plan (Gans/Gansc)  $(date '+%F %T')"
    CELL="Gorig Ga Gnsc Gansc Niso Nnsc" bash ablation.sh plan
    echo; echo "######################## cpu  $(date '+%F %T')"
    bash ablation.sh cpu
    echo; echo "######################## threads  $(date '+%F %T')"
    bash ablation.sh threads
    echo "끝 $(date '+%F %T')"
  } 2>&1 | tee "$log"
  echo; echo "전체 로그: $log"
  ;;

# ---------------------------------------------------------------- cpu
# CPU 팔을 직접 잰다 (KAIROX_PROFILE_CPU). 지금까지 프로파일링이 CUDA 만 봤다.
#
# CPU 희소 연산은 kairox_executor 워커에 비동기 제출되고 split_fut.get() 으로 합류한다.
# 그 블로킹 구간들의 경과 시간을 재면 두 팔의 선후가 그대로 나온다.
#   work   워커가 계산한 시간          CPU 팔의 크기
#   join   메인이 CPU 를 기다린 시간   CPU 가 임계 팔일 때의 노출된 비용
#   evsync CPU 가 GPU 를 기다린 시간   GPU 가 앞서 있다는 뜻
# 경과 시간이라 ggml 풀의 스핀 대기에 오염되지 않는다 — perf 로는 못 쟀던 이유다.
cpu)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  CCELLS=${CCELLS:-"Gorig Ga Gnsc Gansc Niso Nnsc"}
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      [[ " $CCELLS " == *" $cn "* ]] || continue
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || continue
      echo "-- $nm / $cn   B=$C_B tau=$C_TAU alpha=$C_AL"
      PROF_CPU=1 WRAP= \
        run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
            "$C_TAU" "$N" 1 0 0 0 > "abl_logs/cpu__${nm}__${cn}.log" 2>&1
      grep -E 'CPU 팔 프로파일|work|join|evsync|판정' "abl_logs/cpu__${nm}__${cn}.log" | sed 's/^/   /'
    done
  done
  ;;

# ---------------------------------------------------------------- threads
# CPU 용량을 바꿔가며 셀별 민감도를 본다.  질문 두 개를 한 번에 답한다.
#
#   (a) CPU 가 임계 경로인가       곡선이 평평하면 아니다
#   (b) 입도가 CPU 작업을 바꾸나   그룹 셀과 뉴런 셀의 기울기가 다르면 그렇다
#
# 서버는 nproc 24 이고 기본 -t 12 는 절반이다. 아래로도 위로도 봐야 한다 —
# 올려서 안 변하면 CPU 가 숨어 있는 게 확정되고, 내려서 뉴런 셀이 덜 나빠지면
# 입도가 CPU 를 비웠다는 직접 증거다. 적중률 장부가 아니라 벽시계로 재는 유일한 길.
#
# CPU 쪽 직접 타이머가 없는 이유: CPU 연산은 backend_cpu 가 ggml 스케줄러 분할로
# 돌아 한 지점을 감쌀 수 없다. cpu_sparse 버킷은 CPU 를 안 재는 것으로 판명됐다.
threads)
  [[ -f "$CAL" ]] || { echo "$CAL 없음 — 먼저 'bash $0 cal'" >&2; exit 1; }
  TLIST=${TLIST:-"2 4 12 24"}
  TCELLS=${TCELLS:-"Ga Gansc Niso Nnsc"}
  declare -A R
  printf '%-13s %-6s %4s %-9s  %s\n' 모델 셀 -t t/s clocks
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 <<<"$e"; gs=$((g1 / g16))
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      [[ " $TCELLS " == *" $cn "* ]] || continue
      setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || { echo "   skip $nm/$cn"; continue; }
      for th in $TLIST; do
        printf '%-13s %-6s %4s ' "$nm" "$cn" "$th"
        v=$(THREADS="$th" WRAP= \
            run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
                "$C_TAU" "$N" 3 0 0 0 \
            | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
        printf '%-9s  %s\n' "${v:-실패}" "$(clocks)"
        [[ -n "$v" ]] && R["$nm|$cn|$th"]=$v
      done
    done
  done
  echo
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm _ <<<"$e"; want "$nm" || continue
    echo "== $nm   t/s (행=셀, 열=스레드).  24/2 가 크면 CPU 민감, 1 에 가까우면 CPU 가 숨어 있다"
    printf '%-7s' 셀; for th in $TLIST; do printf '%8s' "t$th"; done; printf '%9s%9s\n' 24/2 24/12
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      [[ " $TCELLS " == *" $cn "* ]] || continue
      [[ -z "${R["$nm|$cn|${TLIST%% *}"]:-}" ]] && continue
      printf '%-7s' "$cn"
      for th in $TLIST; do printf '%8s' "${R["$nm|$cn|$th"]:-—}"; done
      lo=${R["$nm|$cn|${TLIST%% *}"]:-}; hi=${R["$nm|$cn|${TLIST##* }"]:-}; mid=${R["$nm|$cn|12"]:-}
      printf '%9s%9s\n' \
        "$([[ -n "$lo$hi" ]] && awk -v a="$hi" -v b="$lo" 'BEGIN{printf "%.3f", a/b}' || echo —)" \
        "$([[ -n "$mid$hi" ]] && awk -v a="$hi" -v b="$mid" 'BEGIN{printf "%.3f", a/b}' || echo —)"
    done
    echo
  done
  ;;

# ---------------------------------------------------------------- all
# 전 패밀리를 순서대로. 5 모델이면 일곱 시간쯤 걸리므로 로그를 파일로 남긴다.
# 한 패밀리가 실패해도 다음으로 넘어간다 — 긴 런이 중간에서 통째로 죽지 않게.
# 다만 cal 이 실패하면 나머지가 전부 무의미하므로 거기서는 멈춘다.
# ONLY / CELL / REPS / N 은 환경변수라 그대로 하위 호출에 상속된다.
all)
  log=${ALL_LOG:-abl_all_$(date +%m%d_%H%M).log}
  echo "전체 로그: $log"
  {
    echo "시작 $(date '+%F %T')   모델 ${#MODELS[@]} 개   N=$N NSYS_N=$NSYS_N REPS=$REPS"
    bash ablation.sh show
    for f in cal ts plan cpu nsys; do   # slope 는 5 모델 0.93~1.10 으로 확정, 뺐다
      echo; echo "######################## $f   $(date '+%F %T')"
      t0=$SECONDS
      if bash ablation.sh "$f"; then
        printf '######################## %s 완료  %d 분\n' "$f" $(( (SECONDS - t0) / 60 ))
      else
        printf '######################## %s 실패  %d 분\n' "$f" $(( (SECONDS - t0) / 60 ))
        [[ "$f" == cal ]] && { echo "cal 이 실패하면 나머지가 무의미하다 — 중단"; break; }
      fi
    done
    echo; echo "######################## 요약   $(date '+%F %T')"
    python3 abl_sum.py
    echo "끝 $(date '+%F %T')"
  } 2>&1 | tee "$log"
  echo; echo "전체 로그: $log"
  ;;

*)
  echo "usage: bash ablation.sh {show|quick|all|cal|calg|ts|plan|threads|nsys|slope|dump}" >&2
  exit 1
  ;;
esac
