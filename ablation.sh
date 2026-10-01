#!/bin/bash
# 뉴런 입도 오버헤드의 절제 실험.  opt-6.7b, Bamboo.
#
# 묻는 것: 뉴런 입도의 비용이 어느 커널에 있고, 각 제거 수단이 그중 얼마를 되돌리는가.
# 각 비교가 한 변수만 움직이도록 셀을 짰다.
#
#   Nns   대 Niso    정렬 단독        (NOSORT)
#   Nnsc  대 Nns     호스트 스캔 단독 (GPU_COMPACT)
#   Nb    대 Niso    전송 경로 단독   (cudaMemcpyBatchAsync)
#   Niso  대 G       입도+전송 단독   (같은 전송량, 같은 alpha)
#   G     대 Ga      되먹임 단독      (그룹 입도)
#   N     대 Niso    상한 단독
#
# d_nooverhead.sh 는 NOSORT 와 GPU_COMPACT 를 한 플래그로 묶어 둬서 둘의 효과가 섞였다.
# 여기서는 독립 인자다.
#
# 배포본(alpha=0.05)은 되먹임이 예산을 움직이므로 고정 상한이 의미가 없다.
# 그래서 기준 축(Ga/Na)은 상한 없는 셀에만 두고, 제거 사다리는 alpha=0 에서만 돌린다.
#
# usage
#   bash ablation.sh cal     # B_iso + tau 이분 탐색 -> abl_cal.txt   (~15분)
#   bash ablation.sh ts      # 벽시계 9셀 x 2모델 x 3반복, 라운드로빈 (~35분)
#   bash ablation.sh plan    # PROFILE_PLAN  9셀 x 2모델 x 1런        (~15분)
#   bash ablation.sh nsys    # 커널 분해     9셀 x 2모델 x 1런        (~30분)
#   bash ablation.sh slope   # PLAN_DELAY_US 0/200/400 on Niso        (~3분)
#   bash ablation.sh dump    # 품질 지표(낭비율/적중률)               (~20분)
#
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
N=${N:-128}
CAL=${CAL:-abl_cal.txt}
BIN=./build_rel/bin/llama-completion

# 이름|모델|split 접두|그룹 n_group|g=1 n_group|group_size
MODELS=(
  "opt-6.7b|opt-6.7b|opt-6.7b-sparkinfer-model-split|1024|16384|16"
  "Bamboo|Bamboo-base-v0_1|Bamboo-base-v0_1-sparkinfer-model-split|896|14336|16"
)

# 이름 split zc gather batch 예산 nosort compact alpha
CELLS=(
  "Ga    g16 0 0 0 one  0 0 0.05"
  "Na    g1  1 0 0 one  0 0 0.05"
  "G     g16 0 0 0 one  0 0 0.00"
  "N     g1  1 0 0 one  0 0 0.00"
  "Niso  g1  1 0 0 iso  0 0 0.00"
  "Nhalf g1  1 0 0 half 0 0 0.00"
  "Nns   g1  1 0 0 iso  1 0 0.00"
  "Nnsc  g1  1 0 0 iso  1 1 0.00"
  "Nb    g1  0 0 1 iso  0 0 0.00"
)

[[ -x "$BIN" ]] || { echo "MISSING $BIN — bash compile_kairox.sh rel" >&2; exit 1; }
mkdir -p prof abl_logs abl_dumps

want()  { [[ -z "$ONLY" || "$1" == *"$ONLY"* ]]; }
wantc() { [[ -z "$CELL" || "$1" == "$CELL" ]]; }
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
    ${WRAP:-} $BIN -m "$M/$1.gguf" -kairox-ms "$M/$2.gguf" \
      -cffn -fit off -ngl all --no-mmap --no-direct-io -vb 0 -no-cnv \
      --repeat-penalty 1.1 -t 12 -s 42 -c 1024 -n "${11}" --no-warmup --ignore-eos \
      --bench-prompt-file prompts.txt --bench-runs "${12}" --bench-warmup 0 --bench-no-print 2>&1
}

# 셀 한 줄을 풀어 run 인자로 쓸 전역을 채운다.
setcell() {
  local nm=$1 mo=$2 base=$3 g16=$4 g1=$5
  read -r C_NAME C_SPL C_ZC C_GA C_BA C_BUD C_NS C_CP C_AL <<<"$6"
  C_SPLIT="$base-$g16"; [[ "$C_SPL" == g1 ]] && C_SPLIT="$base-$g1"
  C_TAU=0
  if ((C_NS)); then
    C_TAU=$(cal_get "$nm" tau)
    [[ -z "$C_TAU" ]] && return 1
  fi
  case "$C_BUD" in
    one)  C_B=1.0 ;;
    iso)  C_B=$(cal_get "$nm" iso) ;;
    half) C_B=$(awk -v x="$(cal_get "$nm" iso)" 'BEGIN{if(x!="")printf "%.6f", x/2}') ;;
  esac
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
# B_iso: 그룹 셀 G 가 실제로 실행한 전송량(뉴런/레이어-스텝)에 g=1 을 맞춘다.
#   그룹 셀의 "실행" 은 그룹 단위이므로 x group_size 가 뉴런 수다.
#   K 로 계산하지 않는다 — 덤프 K 와 PROFILE_PLAN K 가 어긋나 있고, 여기선 실행량이 직접 보인다.
# tau: NOSORT 는 |S| 를 K 에 맞추는 제어가 없다. |S|/K -> 1 이 되는 tau 를 찾는다.
cal)
  : > "$CAL"
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
    want "$nm" || continue
    check "$nm" "$mo" "$base" "$g16" "$g1" || continue

    echo "== $nm : 그룹 기준 전송량 측정 (G, 상한 없음)"
    gref=$(run "$mo" "$base-$g16" 0 0 0 1.0 0 0 0.00 0 64 1 1 0 0 | pairs_of)
    if [[ -z "$gref" ]]; then echo "   실패 — PROFILE_PLAN 출력 없음 (KAIROX_PARALLEL 확인)"; continue; fi
    nref=$(awk -v p="$gref" -v g="$gs" 'BEGIN{printf "%.2f", p*g}')
    printf '   그룹 실행 %s 짝/레이어-스텝  x g=%s  ->  목표 %s 뉴런/레이어-스텝\n' "$gref" "$gs" "$nref"
    echo "$nm gref $gref" >> "$CAL"
    echo "$nm nref $nref" >> "$CAL"

    echo "== $nm : B_iso 이분 탐색 (g=1 의 실행량을 $nref 에 맞춘다 — B 에 단조 증가)"
    lo=0.0002; hi=1.0; best=; bestd=999; bestv=
    for it in 1 2 3 4 5 6 7 8; do
      b=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.6f", sqrt(a*c)}')
      v=$(run "$mo" "$base-$g1" 1 0 0 "$b" 0 0 0.00 0 64 1 1 0 0 | pairs_of)
      [[ -z "$v" ]] && { printf '   B=%-9s (측정 실패)\n' "$b"; break; }
      r=$(awk -v x="$v" -v t="$nref" 'BEGIN{printf "%.3f", x/t}')
      printf '   B=%-9s 실행 %-8s  실행/목표=%s\n' "$b" "$v" "$r"
      d=$(awk -v x="$r" 'BEGIN{d=x-1; if(d<0)d=-d; print d}')
      awk -v a="$d" -v c="$bestd" 'BEGIN{exit !(a<c)}' && { best=$b; bestd=$d; bestv=$r; }
      awk -v a="$bestd" 'BEGIN{exit !(a<0.02)}' && break
      if awk -v x="$r" 'BEGIN{exit !(x>1)}'; then hi=$b; else lo=$b; fi
    done
    if [[ -n "$best" ]]; then
      echo "$nm iso $best" >> "$CAL"; echo "$nm isorat $bestv" >> "$CAL"
      f=""; awk -v x="$bestv" 'BEGIN{exit !(x<0.95 || x>1.05)}' && f="   <<< 5% 밖 — 동일전송 비교 불가"
      echo "   -> B_iso=$best  실행/목표=$bestv$f"
    fi

    echo "== $nm : tau 이분 탐색 (|S|/K -> 1, |S| 는 tau 에 단조 감소)"
    lo=0.0005; hi=0.8; best=; bestd=999; bestv=
    for it in 1 2 3 4 5 6 7; do
      t=$(awk -v a="$lo" -v c="$hi" 'BEGIN{printf "%.5f", sqrt(a*c)}')
      r=$(run "$mo" "$base-$g1" 1 0 0 1.0 1 0 0.00 "$t" 64 1 1 0 0 | sel_of)
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
  printf '%-10s %-6s %-6s %-9s %-6s %5s  %-9s  %s\n' 모델 셀 반복 B tau rep t/s clocks
  for ((rep=1; rep<=REPS; rep++)); do
    for e in "${MODELS[@]}"; do
      IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
      want "$nm" || continue
      check "$nm" "$mo" "$base" "$g16" "$g1" >/dev/null || continue
      for c in "${CELLS[@]}"; do
        read -r cn _ <<<"$c"
        wantc "$cn" || continue
        setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c" || { echo "   skip $nm/$cn — $CAL 에 값 없음"; continue; }
        ck=$(clocks)
        printf '%-10s %-6s %-6s %-9s %-6s %5s  ' "$nm" "$cn" "$C_BUD" "$C_B" "$C_TAU" "$rep"
        v=$(WRAP= run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
                   "$C_TAU" 512 3 0 0 0 \
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
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
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
  for e in "${MODELS[@]}"; do
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
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
                 "$C_TAU" "$N" 1 0 0 0 \
          | grep -oE 'decode mean:[[:space:]]*[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
      echo "$nm $cn ${w:-NA} $(clocks)" >> abl_nsys_wall.txt
      echo "   짝 벽시계 ${w:-실패} t/s"
      WRAP="nsys profile --force-overwrite=true -o prof/$tag --trace=cuda --sample=none --cpuctxsw=none" \
        run "$mo" "$C_SPLIT" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_NS" "$C_CP" "$C_AL" \
            "$C_TAU" "$N" 1 0 0 0 > "abl_logs/nsys__${nm}__${cn}.log" 2>&1
      [[ -f "prof/$tag.nsys-rep" ]] || { echo "   리포트 생성 실패 — 로그 확인"; continue; }
      nsys stats --force-export=true --format csv -o "prof/$tag" \
        --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
        "prof/$tag.nsys-rep" > "abl_logs/stats__${nm}__${cn}.log" 2>&1
      ls "prof/${tag}"_*.csv 2>/dev/null | sed 's/^/   /'
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
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
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
                   "$C_TAU" 512 3 0 0 "$us" \
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
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
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
    IFS='|' read -r nm mo base g16 g1 gs <<<"$e"
    want "$nm" || continue
    for c in "${CELLS[@]}"; do
      read -r cn _ <<<"$c"
      wantc "$cn" || continue
      if setcell "$nm" "$mo" "$base" "$g16" "$g1" "$c"; then
        printf '%-10s %-6s %-26s %-4s %-4s %-4s %-9s %-8s %-7s %-8s %s\n' \
          "$nm" "$C_NAME" "...-${C_SPLIT##*-}" "$C_ZC" "$C_GA" "$C_BA" "$C_B" "$C_TAU" \
          "$C_NS" "$C_CP" "$C_AL"
      else
        printf '%-10s %-6s  <<< %s 에 값 없음 (cal 먼저)\n' "$nm" "$cn" "$CAL"
      fi
    done
  done
  ;;

*)
  echo "usage: bash ablation.sh {show|cal|ts|plan|nsys|slope|dump}" >&2
  exit 1
  ;;
esac
