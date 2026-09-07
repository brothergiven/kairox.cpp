// KAIROX reload 경로의 전송 비용을 분해하는 마이크로벤치마크.
//
// group_size 스윕에서 decode 시간이 전송 "바이트"가 아니라 "호출 횟수"에 지배된다는
// 결과가 나왔다(호출당 약 16.9 us). 이 벤치마크는 그 값을 추론이 아니라 직접 측정하고,
// gather 버퍼가 실제로 얼마나 회수해 주는지 확인한다.
//
// 측정 항목
//   A. 전송 크기 x 동기화 주기 -> 호출당 고정비용과 실효 대역폭
//   B. 흩어진 행 N개를 개별 전송 vs pinned staging buffer 로 gather 후 1회 전송
//
// 빌드: nvcc -O3 -o bench_pcie bench_pcie.cu
//
// 기본값은 prosparse-llama-2-7b 형상에 맞춰져 있다.
// 뉴런 1개(row) = 4352 B (Q8_0, n_embd=4096), 그룹 16개 = 69632 B.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        cudaError_t err_ = (x);                                                          \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                 \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static constexpr size_t kRowBytes = 4352;  // Q8_0, n_embd=4096 기준 뉴런 1개

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// A. 전송 크기와 동기화 주기를 바꿔가며 호출당 비용을 잰다.
//    sync_every = 0 이면 전부 큐에 넣고 마지막에 한 번만 동기화한다.
static void bench_transfer(char * h_src, char * d_dst, size_t src_bytes, size_t dst_bytes) {
    printf("\n=== A. 전송 크기 x 동기화 주기 ===\n");
    printf("%10s %8s %12s %12s %12s\n", "크기(KB)", "sync주기", "호출당(us)", "대역폭(GB/s)", "총시간(ms)");

    const size_t sizes[]      = { 4352, 8704, 17408, 34816, 69632, 139264, 278528, 1048576, 4194304 };
    const int    sync_every[] = { 1, 4, 16, 0 };
    const size_t total_bytes  = 512ull << 20;  // 설정마다 512 MiB 를 옮긴다

    for (size_t sz : sizes) {
        for (int se : sync_every) {
            const size_t n = std::min(total_bytes / sz, dst_bytes / sz);
            if (n == 0) continue;

            // 워밍업
            for (size_t i = 0; i < std::min<size_t>(n, 16); ++i) {
                CUDA_CHECK(cudaMemcpyAsync(d_dst + (i % (dst_bytes / sz)) * sz,
                                           h_src + (i % (src_bytes / sz)) * sz, sz,
                                           cudaMemcpyHostToDevice, 0));
            }
            CUDA_CHECK(cudaStreamSynchronize(0));

            const double t0 = now_ms();
            for (size_t i = 0; i < n; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(d_dst + (i % (dst_bytes / sz)) * sz,
                                           h_src + (i % (src_bytes / sz)) * sz, sz,
                                           cudaMemcpyHostToDevice, 0));
                if (se > 0 && (i + 1) % se == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(0));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
            const double dt = now_ms() - t0;

            printf("%10.1f %8s %12.2f %12.2f %12.1f\n", sz / 1024.0,
                   se ? std::to_string(se).c_str() : "끝에만",
                   dt * 1000.0 / n, (double) n * sz / (dt / 1e3) / 1e9, dt);
        }
    }
}

// B. 흩어진 행을 개별 전송 vs gather 후 일괄 전송.
//    KAIROX 의 reload 는 전자에 해당하고, 제안하는 구조는 후자다.
static void bench_gather(char * h_weights, char * h_stage, char * d_dst,
                         size_t n_rows_total, size_t dst_bytes) {
    printf("\n=== B. 개별 전송 vs gather 후 일괄 전송 ===\n");
    printf("행 1개 = %zu B (뉴런 1개). 흩어진 행을 무작위로 골라 옮긴다.\n", kRowBytes);
    printf("%10s %14s %14s %12s %12s %8s\n",
           "행 수", "개별(ms)", "gather+1회(ms)", "gather(ms)", "전송(ms)", "배속");

    std::mt19937 rng(42);
    for (size_t n_rows : { 64, 256, 1024, 4096, 16384 }) {
        if (n_rows * kRowBytes > dst_bytes) break;

        std::vector<size_t> idx(n_rows);
        std::uniform_int_distribution<size_t> dist(0, n_rows_total - 1);
        for (auto & v : idx) v = dist(rng);

        // (1) 개별 전송 — 현재 kairox_batch_reload 구조 (4개마다 동기화)
        CUDA_CHECK(cudaStreamSynchronize(0));
        double t0 = now_ms();
        for (size_t i = 0; i < n_rows; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(d_dst + i * kRowBytes, h_weights + idx[i] * kRowBytes,
                                       kRowBytes, cudaMemcpyHostToDevice, 0));
            if ((i + 1) % 4 == 0) CUDA_CHECK(cudaStreamSynchronize(0));
        }
        CUDA_CHECK(cudaStreamSynchronize(0));
        const double t_individual = now_ms() - t0;

        // (2) gather 후 일괄 전송
        t0 = now_ms();
        for (size_t i = 0; i < n_rows; ++i) {
            memcpy(h_stage + i * kRowBytes, h_weights + idx[i] * kRowBytes, kRowBytes);
        }
        const double t_gather = now_ms() - t0;

        t0 = now_ms();
        CUDA_CHECK(cudaMemcpyAsync(d_dst, h_stage, n_rows * kRowBytes, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
        const double t_xfer = now_ms() - t0;

        printf("%10zu %14.3f %14.3f %12.3f %12.3f %7.1fx\n", n_rows, t_individual,
               t_gather + t_xfer, t_gather, t_xfer, t_individual / (t_gather + t_xfer));
    }
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s\n", prop.name);

    const size_t src_bytes   = 512ull << 20;  // pinned 소스 (CPU 상주 가중치 역할)
    const size_t stage_bytes = 128ull << 20;  // pinned staging buffer
    const size_t dst_bytes   = 256ull << 20;  // GPU 캐시 역할

    char *h_src = nullptr, *h_stage = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaHostAlloc((void **) &h_src, src_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &h_stage, stage_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc((void **) &d_dst, dst_bytes));
    memset(h_src, 1, src_bytes);

    bench_transfer(h_src, d_dst, src_bytes, dst_bytes);
    bench_gather(h_src, h_stage, d_dst, src_bytes / kRowBytes, dst_bytes);

    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_stage));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}
