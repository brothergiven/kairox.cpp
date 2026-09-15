#!/bin/bash
# 새 장비 스펙 확인 + 배치 실험 설정 추천.
# VRAM 계산 상수는 prosparse-llama-2-7b Q8_0 (-cffn) 실측값이다. 다른 모델이면 로그로 다시 잴 것.
#
# usage: [N=512] [CACHE_MIB=1364] bash hw_check.sh

set -uo pipefail

N=${N:-512}
CACHE_MIB=${CACHE_MIB:-1364}   # KAIROX FFN 캐시 목표. 1364 = 3070 배치 실험(112,880 뉴런)과 같은 용량
MODEL_GPU_MIB=3075             # 로그 "CUDA0 model buffer size"
COMPUTE_MIB=170                # 로그 "CUDA0 compute buffer size" (CTX 에 따라 약간 변함)
MARGIN_MIB=512                 # llama-kairox.cpp 의 vram_budget 여유분
KV_KIB_PER_TOK=512             # f16: 32 레이어 x K/V x 4096 차원 x 2 바이트 = 512 KiB/토큰
PROMPT_TOK=43                  # 배치 스크립트 기본 프롬프트 어림
OVERHEAD_MIB=700               # VB 밖에서 쓰는 것: CUDA 컨텍스트 + gather staging(64) + 여유

echo "== CPU"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L3 cache"
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
echo "  물리 코어 $phys / 논리 코어 $(nproc)"

echo
echo "== RAM"
free -g | sed -n 1,2p
avail_gib=$(free -g | awk '/^Mem:/ { print $7 }')

echo
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,memory.free,pcie.link.gen.current,pcie.link.width.current,driver_version \
    --format=csv
command -v nvcc >/dev/null && nvcc --version | tail -1
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)

echo
echo "== 추천"
threads=$(( phys / 2 ))
(( threads < 2 )) && threads=2
echo "  THREADS=$threads   (물리 코어 절반. ggml 스레드 + 워커 1 + 메인 1 이 물리 코어를 넘지 않게)"
(( avail_gib >= 14 )) || echo "  !! 가용 RAM ${avail_gib}GiB — --no-mmap 으로 9GB 모델을 올리면 swap 위험 (14GiB 이상 권장)"
(( gpu_procs == 0 )) || echo "  !! GPU 를 쓰는 프로세스 ${gpu_procs}개 — 실험 중에는 모두 끌 것"

echo
echo "  캐시 목표 ${CACHE_MIB} MiB, N=$N 에서 최대 np 별 설정 (모든 np 가 같은 CTX/VB 를 쓴다):"
printf "  %-7s %8s %7s %4s %11s  %s\n" "np_max" "KV토큰" "CTX" "VB" "GPU필요MiB" "이 장비"
for np in 4 8 16 24 32 48; do
    need=$(( PROMPT_TOK + (N - PROMPT_TOK) * np ))
    ctx=$(( (need + 255) / 256 * 256 ))
    kv_mib=$(( ctx * KV_KIB_PER_TOK / 1024 ))
    vb_mib=$(( MODEL_GPU_MIB + kv_mib + COMPUTE_MIB + MARGIN_MIB + CACHE_MIB ))
    vb=$(( (vb_mib + 1023) / 1024 ))
    gpu_need=$(( vb * 1024 + OVERHEAD_MIB ))
    ok=$( (( gpu_need <= gpu_free )) && echo "가능" || echo "불가" )
    printf "  %-7s %8d %7d %4d %11d  %s%s\n" "$np" "$need" "$ctx" "$vb" "$gpu_need" "$ok" \
        "$( (( np > 8 )) && echo '  (np>8: GPU dense 경로)' )"
done

echo
echo "  np > 8 이면 build_sparse_ffn 이 GPU 에서 sparse 대신 dense matmul + scatter 를 쓴다."
echo "  np 8 이하와 초과는 다른 영역으로 따로 표시할 것."