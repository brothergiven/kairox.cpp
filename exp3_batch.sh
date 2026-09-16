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
NPS="1 2 4 8 16"
PAIRS="16:0 16:1 1:1"
ROUNDS=5
OUT=./batch_bamboo
MODEL=$HOME/SPIF-GGUF/Bamboo-base-v0_1.gguf
SPLIT=$HOME/SPIF-GGUF/Bamboo-base-v0_1-sparkinfer-model-split
PROMPT="Explain how operating systems manage virtual memory, including paging, page tables, TLBs, and what happens on a page fault."
# ----------------------------------------------

mkdir -p "$OUT"
for r in $(seq 1 $ROUNDS); do
  for np in $NPS; do
    for p in $PAIRS; do
      g=${p%%:*}; gather=${p##*:}
      log=$OUT/np${np}_g${g}_gather${gather}_r${r}.log
      grep -q "decoded .* tokens in" "$log" 2>/dev/null && continue   # 이미 끝난 런은 건너뜀
      echo "[run] $(basename $log .log)  $(date +%T)"
      # KAIROX_PARALLEL=1 필수(없으면 재배치 꺼짐), -kvu 필수(없으면 배치 디코드 실패)
      env KAIROX_PARALLEL=1 KAIROX_GATHER=$gather \
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
  END { for (k in s) { split(k, a, "_"); printf "%-4s %-7s %8.2f %4d %12d\n",
        substr(a[1],3), substr(a[2],2) ":" substr(a[3],7), s[k]/n[k], n[k], c[k] } }' | sort -k1,1n -k2,2
