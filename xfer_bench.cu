// KAIROX 전송 경로 마이크로벤치: host staging gather vs GPU zero-copy(워드 스레드).
//   C) base        : 원본 경로 — 그룹마다 cudaMemcpyAsync 한 번, window(4) 마다 동기화
//   A) host gather : CPU memcpy -> pinned staging -> cudaMemcpyAsync -> scatter 커널
//   B) zero-copy   : GPU 커널이 pinned host 를 직접 읽어 캐시 슬롯에 씀
// 실제 호출 모양을 흉내낸다: 흩어진 그룹 -> 흩어진 슬롯.
//
// build: nvcc -O3 -arch=native -o gather_bench gather_bench.cu
// run  : ./gather_bench [row_bytes] [rows_per_group]     (기본 8192 = F16 4096열, 1)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

__global__ void scatter_k(const uint4* __restrict__ src, uint4* __restrict__ dst,
                          const int* __restrict__ slot, int elems) {
    const int g = blockIdx.x;
    const uint4* s = src + (size_t)g*elems;
    uint4* d = dst + (size_t)slot[g]*elems;
    for (int i = threadIdx.x; i < elems; i += blockDim.x) d[i] = s[i];
}
// 워드 스레드: 전체 워드에 스레드를 1:1 로 깔아 PCIe 요청을 최대한 겹친다.
__global__ void zerocopy_k(const ulonglong2* __restrict__ host_base, ulonglong2* __restrict__ cache_base,
                           const int* __restrict__ grp, const int* __restrict__ slot,
                           int words_per_group, int n_groups) {
    size_t tid = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t total = (size_t)words_per_group*n_groups;
    for (; tid < total; tid += (size_t)gridDim.x*blockDim.x) {
        int g = tid / words_per_group, w = tid % words_per_group;
        cache_base[(size_t)slot[g]*words_per_group + w] = host_base[(size_t)grp[g]*words_per_group + w];
    }
}

int main(int argc, char** argv) {
    const int row_bytes = argc > 1 ? atoi(argv[1]) : 8192;   // F16 4096열 = 8192B (Q8_0 이면 4352)
    const int rows_per_group = argc > 2 ? atoi(argv[2]) : 1; // g
    const size_t gnb = (size_t)row_bytes * rows_per_group;   // 그룹당 바이트
    const int n_total_groups = 4096, n_cache_slots = 2048;

    char *weights, *stage_h; char* cache_d; char* stage_d; int *slot_h,*slot_d,*grp_h,*grp_d;
    CK(cudaHostAlloc((void**)&weights, gnb*n_total_groups, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&stage_h, gnb*1024, cudaHostAllocDefault));
    CK(cudaMalloc((void**)&stage_d, gnb*1024));
    CK(cudaMalloc((void**)&cache_d, gnb*n_cache_slots));
    CK(cudaHostAlloc((void**)&slot_h, 1024*sizeof(int), cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&grp_h, 1024*sizeof(int), cudaHostAllocDefault));
    CK(cudaMalloc((void**)&slot_d, 1024*sizeof(int))); CK(cudaMalloc((void**)&grp_d, 1024*sizeof(int)));
    for (size_t i=0;i<gnb*n_total_groups;i+=4096) weights[i]=(char)i;

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::mt19937 rng(42);

    printf("row_bytes=%d g=%d  그룹당 %.1f KiB\n", row_bytes, rows_per_group, gnb/1024.0);
    printf("%8s %10s %12s %12s %12s %9s %9s\n", "groups", "MiB", "C base(us)", "A host(us)", "B zc(us)", "A/C", "B/C");
    for (int ng : {4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        std::vector<int> gs(n_total_groups), ss(n_cache_slots);
        for (int i=0;i<n_total_groups;i++) gs[i]=i; for (int i=0;i<n_cache_slots;i++) ss[i]=i;
        std::shuffle(gs.begin(), gs.end(), rng); std::shuffle(ss.begin(), ss.end(), rng);
        for (int i=0;i<ng;i++){ grp_h[i]=gs[i]; slot_h[i]=ss[i]; }
        CK(cudaMemcpy(grp_d,grp_h,ng*sizeof(int),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(slot_d,slot_h,ng*sizeof(int),cudaMemcpyHostToDevice));
        const double mib = gnb*(double)ng/1048576.0;
        float ta=0, tb=0, tc=0; const int REP=20;
        const int WINDOW=4;
        for (int r=0;r<REP+3;r++) {              // C: 원본 — 그룹당 memcpy, window 마다 sync
            if (r==3) CK(cudaEventRecord(e0,st));
            for (int off=0; off<ng; off+=WINDOW) {
                int n = std::min(WINDOW, ng-off);
                for (int i=off;i<off+n;i++)
                    CK(cudaMemcpyAsync(cache_d+(size_t)slot_h[i]*gnb, weights+(size_t)grp_h[i]*gnb, gnb, cudaMemcpyHostToDevice, st));
                CK(cudaStreamSynchronize(st));
            }
        }
        CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&tc,e0,e1));


        for (int r=0;r<REP+3;r++) {              // A: host staging gather
            if (r==3) CK(cudaEventRecord(e0,st));
            for (int i=0;i<ng;i++) memcpy(stage_h+(size_t)i*gnb, weights+(size_t)grp_h[i]*gnb, gnb);
            CK(cudaMemcpyAsync(stage_d, stage_h, gnb*ng, cudaMemcpyHostToDevice, st));
            scatter_k<<<ng, 256, 0, st>>>((const uint4*)stage_d,(uint4*)cache_d, slot_d, (int)(gnb/16));
            CK(cudaStreamSynchronize(st));
        }
        CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ta,e0,e1));

        const int wpg = (int)(gnb/16); const int thr=1024;
        for (int r=0;r<REP+3;r++) {              // B: zero-copy
            if (r==3) CK(cudaEventRecord(e0,st));
            int blocks = (int)std::min<size_t>(65535, ((size_t)wpg*ng + thr-1)/thr);
            zerocopy_k<<<blocks, thr, 0, st>>>((const ulonglong2*)weights,(ulonglong2*)cache_d, grp_d, slot_d, wpg, ng);
            CK(cudaStreamSynchronize(st));
        }
        CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&tb,e0,e1));
        printf("%8d %10.2f %12.1f %12.1f %12.1f %8.2fx %8.2fx\n", ng, mib,
               tc*1000/REP, ta*1000/REP, tb*1000/REP, tc/ta, tc/tb);
    }
    return 0;
}
