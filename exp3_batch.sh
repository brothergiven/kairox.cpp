#!/bin/bash
# 배치 크기(np) x (g:gather) 처리량 비교 — 설정은 아래를 직접 고친다.
#   16:0 base(원본 전송) / 16:1 B1(전송 병합) / 1:1 ours(전송 병합 + 뉴런 단위 입도)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# ---- 설정 (hw_check.sh 결과 보고 직접 정하기) ----
THREADS=6                       # 물리 코어 절반 전후, THREADS+2 <= 물리 코어
VB=9                            # VB*1024 + 700 <= GPU free (MiB)
CTX=7680                        # >= 43 + (N-43) * 최대 np. 모든 np 가 같은 값을 써야 캐시 용량이 같다
N=512
NPS="1 4 8 16"
PAIRS="16:0 16:1 16:2 1:1 1:2"   # g:전송경로 (0=원본 memcpy, 1=gather, 2=zerocopy). g=1+원본은 너무 느려 제외
ROUNDS=3
OUT=./batch_bamboo
MODEL=$HOME/SPIF-GGUF/Bamboo-base-v0_1.gguf
SPLIT=$HOME/SPIF-GGUF/Bamboo-base-v0_1-sparkinfer-model-split
PROMPT="Explain how operating systems manage virtual memory, including paging, page tables, TLBs, and what happens on a page fault."
# ----------------------------------------------

mkdir -p "$OUT"
for r in $(seq 1 $ROUNDS); do
  for np in $NPS; do
    for p in $PAIRS; do
      g=${p%%:*}; path=${p##*:}
      log=$OUT/np${np}_g${g}_p${path}_r${r}.log
      grep -q "decoded .* tokens in" "$log" 2>/dev/null && continue
      echo "[run] $(basename $log .log)  $(date +%T)"
      env KAIROX_PARALLEL=1 \
          KAIROX_GATHER=$([ "$path" = 1 ] && echo 1 || echo 0) \
          KAIROX_ZEROCOPY=$([ "$path" = 2 ] && echo 1 || echo 0) \
        ./build_rel/bin/llama-batched \
          -m "$MODEL" -kairox-ms "$SPLIT-$(( 14336 / g )).gguf" \
          -cffn -fit off -ngl all --no-mmap --no-direct-io -vb $VB -t $THREADS \
          -c $CTX -kvu -np $np -n $N --ignore-eos -s 42 -p "$PROMPT" \
          > "$log" 2>&1 || echo "  !! 실패: $log"
    done
  done
done

# ---- 요약: 조건별 평균 t/s, 그리고 캐시 용량(np 사이에 같아야 함) ----
echo
printf "%-4s %-7s %8s %4s %12s\n" np "g:gat" "t/s" n "cached(sum)"
for f in $OUT/np*_r*.log; do
  t=$(grep -o "speed: *[0-9.]*" "$f" | awk '{print $2}')
  c=$(grep -o "cached *[0-9]*" "$f" | awk '{s+=$2} END{print s+0}')
  [ -n "$t" ] && echo "$(basename $f .log | sed 's/_r[0-9]*$//') $t $c"
done | awk '{ s[$1]+=$2; n[$1]++; c[$1]=$3 }
  END { for (k in s) { split(k, a, "_"); printf "%-4s %-8s %8.2f %4d %12d\n",
        substr(a[1],3), substr(a[2],2) ":" substr(a[3],2), s[k]/n[k], n[k], c[k] } }' | sort -k1,1n -k2,2



# ---- np 별 배율 (원본 전송 g=16 대비) ----
echo
printf "%-4s %10s %10s %10s %10s\n" np "16:gather" "16:zc" "1:gather" "1:zc"
for f in $OUT/np*_r*.log; do
  t=$(grep -o "speed: *[0-9.]*" "$f" | awk '{print $2}')
  [ -n "$t" ] && echo "$(basename $f .log) $t"
done | awk '{ split($1,a,"_"); np=substr(a[1],3); k=substr(a[2],2) ":" substr(a[3],2)
      s[np,k]+=$2; n[np,k]++; nps[np]=1 }
  function m(np,k) { return n[np,k] ? s[np,k]/n[np,k] : 0 }
  function r(x,y)  { return (x&&y) ? sprintf("%.3fx", x/y) : "-" }
  END { for (np in nps) { b=m(np,"16:0")
        printf "%-4s %10s %10s %10s %10s\n", np, r(m(np,"16:1"),b), r(m(np,"16:2"),b), r(m(np,"1:1"),b), r(m(np,"1:2"),b) } }' | sort -n