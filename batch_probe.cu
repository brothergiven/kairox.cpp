// KAIROX_MEMCPY_BATCH 선검사.
//
// cudaMemcpyBatchAsync 는 버전마다 시그니처가 다르다
// (cudaTypedefs.h 의 PFN_cuMemcpyBatchAsync_v12080 / _v13000):
//   12.8 ~ 12.x : (..., numAttrs, size_t * failIdx, stream)
//   13.0 ~      : (..., numAttrs,                   stream)
// kairox-gather.cu 와 똑같은 분기를 쓴다. 이게 통과하면 본 빌드도 통과한다.
//
// build: nvcc -O3 -arch=native -o batch_probe batch_probe.cu
// run  : ./batch_probe
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
    printf("CUDART_VERSION = %d\n", CUDART_VERSION);

#if !defined(CUDART_VERSION) || CUDART_VERSION < 12080
    printf("FAIL: CUDA 12.8 미만 — KAIROX_MEMCPY_BATCH 는 zerocopy 로 폴백한다\n");
    return 1;
#else
    const size_t gnb = 8192;
    const int    N = 64, SLOTS = 512;

    char *w = nullptr, *c = nullptr;
    if (cudaHostAlloc((void **) &w, gnb * SLOTS, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void **) &c, gnb * SLOTS) != cudaSuccess) {
        printf("FAIL: 할당 실패\n");
        return 1;
    }
    cudaStream_t st;
    cudaStreamCreate(&st);

    std::vector<void *>       dsts(N);
    std::vector<const void *> srcs(N);
    std::vector<size_t>       sizes(N);
    for (int i = 0; i < N; ++i) {
        dsts[i]  = c + (size_t) (i * 7 % SLOTS) * gnb;
        srcs[i]  = w + (size_t) (i * 13 % SLOTS) * gnb;
        sizes[i] = gnb;
    }

    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx     = 0;
    size_t fail_idx     = 0;
    (void) fail_idx;

#if CUDART_VERSION >= 13000
    const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                 N, &attr, &attr_idx, 1, st);
#else
    const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                 N, &attr, &attr_idx, 1, &fail_idx, st);
#endif
    if (err != cudaSuccess) {
        printf("FAIL: cudaMemcpyBatchAsync -> %s (fail_idx=%zu)\n", cudaGetErrorString(err), fail_idx);
        return 1;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        printf("FAIL: 동기화 실패\n");
        return 1;
    }
    printf("OK: 복사 %d 개를 한 번에 제출했다. KAIROX_MEMCPY_BATCH 를 쓸 수 있다.\n", N);
    return 0;
#endif
}
