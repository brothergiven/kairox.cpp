#include "kairox-gather.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

// staging 버퍼의 연속된 그룹들을 GPU 캐시의 흩어진 슬롯으로 흩뿌린다.
// 블록 하나가 그룹 하나를 담당한다.
template <typename T>
static __global__ void kairox_scatter_kernel(const T * __restrict__ src,
                                             T * __restrict__ dst_base,
                                             const int * __restrict__ slot_idx,
                                             int elems_per_group) {
    const int       g = blockIdx.x;
    const T * __restrict__ s = src + (size_t) g * elems_per_group;
    T * __restrict__       d = dst_base + (size_t) slot_idx[g] * elems_per_group;

    for (int i = threadIdx.x; i < elems_per_group; i += blockDim.x) {
        d[i] = s[i];
    }
}

// 전역 staging 버퍼. 실행기 스레드가 하나뿐이라 락 없이 재사용한다.
// 프로세스 수명 동안 유지한다 — CUDA 컨텍스트 파괴 순서에 얽히지 않도록 해제하지 않는다.
namespace {
struct kairox_gather_stage {
    char * host       = nullptr;  // pinned, gather 대상
    char * dev        = nullptr;  // H2D 도착지
    int *  slot_host  = nullptr;  // pinned, 슬롯 인덱스
    int *  slot_dev   = nullptr;
    size_t bytes      = 0;
    int    max_groups = 0;

    void ensure(size_t need_bytes, int need_groups) {
        if (need_bytes > bytes) {
            if (host) {
                CUDA_CHECK(cudaFreeHost(host));
            }
            if (dev) {
                CUDA_CHECK(cudaFree(dev));
            }
            CUDA_CHECK(cudaHostAlloc((void **) &host, need_bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc((void **) &dev, need_bytes));
            bytes = need_bytes;
        }
        if (need_groups > max_groups) {
            if (slot_host) {
                CUDA_CHECK(cudaFreeHost(slot_host));
            }
            if (slot_dev) {
                CUDA_CHECK(cudaFree(slot_dev));
            }
            CUDA_CHECK(cudaHostAlloc((void **) &slot_host, need_groups * sizeof(int), cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc((void **) &slot_dev, need_groups * sizeof(int)));
            max_groups = need_groups;
        }
    }
};

kairox_gather_stage g_stage;

// 16 바이트 벡터 복사가 가능한지. group_nbytes 와 두 base 주소가 모두 16 정렬이어야 한다.
// cudaMalloc 은 256 정렬을 보장하므로 실질적으로는 cache_base 와 group_nbytes 만 확인하면 된다.
inline bool can_vectorize(const char * cache_base, const char * stage_dev, size_t group_nbytes) {
    return group_nbytes % sizeof(uint4) == 0 && (uintptr_t) cache_base % sizeof(uint4) == 0 &&
           (uintptr_t) stage_dev % sizeof(uint4) == 0;
}
}  // namespace

// scatter 결과를 되읽어 원본 가중치와 바이트 단위로 비교한다.
// 실패 개수를 전역에 누적하고 첫 실패만 한 번 보고한다.
static void kairox_gather_verify_chunk(const char *        weight_base,
                                       const char *        cache_base,
                                       size_t              group_nbytes,
                                       cudaStream_t        stream,
                                       const reload_pair * plan,
                                       size_t              n) {
    static std::vector<char> readback;
    static size_t            n_checked = 0;
    static size_t            n_failed  = 0;
    static bool              reported  = false;

    readback.resize(group_nbytes);

    for (size_t i = 0; i < n; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(readback.data(), cache_base + (size_t) plan[i].slot_idx * group_nbytes,
                                   group_nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        ++n_checked;
        if (memcmp(readback.data(), weight_base + (size_t) plan[i].group_idx * group_nbytes, group_nbytes) != 0) {
            ++n_failed;
            if (!reported) {
                reported = true;
                fprintf(stderr, "KAIROX_GATHER_VERIFY: 불일치 — group=%d slot=%d nbytes=%zu\n", plan[i].group_idx,
                        plan[i].slot_idx, group_nbytes);
            }
        }
    }

    // 진행 상황을 주기적으로 알린다. 전부 통과하면 failed=0 이 계속 찍힌다.
    if (n_checked % 20000 < n) {
        fprintf(stderr, "KAIROX_GATHER_VERIFY: checked=%zu failed=%zu\n", n_checked, n_failed);
    }
}

void kairox_gather_reload(char *              weight_base,
                          char *              cache_base,
                          size_t              group_nbytes,
                          cudaStream_t        stream,
                          const reload_pair * reload_plan,
                          size_t              reload_count) {
    if (reload_count == 0) {
        return;
    }

    const size_t budget = (size_t) std::max(1, k_kairox_gather_budget_mib) << 20;
    // 예산 안에 들어가는 그룹 수. 최소 1개는 처리해야 하므로 group_nbytes 가 예산보다 커도 1로 둔다.
    const size_t chunk = std::max<size_t>(1, budget / group_nbytes);

    for (size_t off = 0; off < reload_count; off += chunk) {
        const size_t n = std::min(chunk, reload_count - off);

        g_stage.ensure(n * group_nbytes, (int) n);

        // 1) host 쪽에서 흩어진 그룹을 연속으로 모은다.
        for (size_t i = 0; i < n; ++i) {
            const reload_pair & p = reload_plan[off + i];
            memcpy(g_stage.host + i * group_nbytes, weight_base + (size_t) p.group_idx * group_nbytes, group_nbytes);
            g_stage.slot_host[i] = p.slot_idx;
        }

        // 2) H2D 2회 — 가중치 한 덩어리 + 슬롯 인덱스.
        CUDA_CHECK(cudaMemcpyAsync(g_stage.dev, g_stage.host, n * group_nbytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(g_stage.slot_dev, g_stage.slot_host, n * sizeof(int), cudaMemcpyHostToDevice, stream));

        // 3) scatter 커널 1회.
        const int threads = 256;
        if (can_vectorize(cache_base, g_stage.dev, group_nbytes)) {
            const int elems = (int) (group_nbytes / sizeof(uint4));
            kairox_scatter_kernel<uint4><<<(int) n, threads, 0, stream>>>(
                (const uint4 *) g_stage.dev, (uint4 *) cache_base, g_stage.slot_dev, elems);
        } else {
            kairox_scatter_kernel<char><<<(int) n, threads, 0, stream>>>(
                (const char *) g_stage.dev, (char *) cache_base, g_stage.slot_dev, (int) group_nbytes);
        }
        CUDA_CHECK(cudaGetLastError());

        // host staging 버퍼를 다음 청크에서 재사용하므로 완료를 기다린다.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (k_kairox_gather_verify) {
            kairox_gather_verify_chunk(weight_base, cache_base, group_nbytes, stream, reload_plan + off, n);
        }
    }
}
