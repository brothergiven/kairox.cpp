#include "kairox-gather.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

// staging 버퍼의 연속된 그룹들을 GPU 캐시의 흩어진 슬롯으로 흩뿌린다. 블록 하나가 그룹 하나를 담당한다.
template <typename T>
static __global__ void kairox_scatter_kernel(const T * __restrict__ src,
                                             T * __restrict__ dst_base,
                                             const int * __restrict__ slot_idx,
                                             int elems_per_group) {
    const int              g = blockIdx.x;
    const T * __restrict__ s = src + (size_t) g * elems_per_group;
    T * __restrict__       d = dst_base + (size_t) slot_idx[g] * elems_per_group;

    for (int i = threadIdx.x; i < elems_per_group; i += blockDim.x) {
        d[i] = s[i];
    }
}

// 전역 staging 버퍼. 실행기 스레드가 하나뿐이라 락 없이 재사용한다.
// CUDA 컨텍스트 파괴 순서에 얽히지 않도록 프로세스 수명 동안 해제하지 않는다.
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

// 16 바이트 벡터 복사가 가능한지. cudaMalloc 은 정렬을 보장하므로 사실상 cache_base 와 group_nbytes 만 본다.
inline bool can_vectorize(const char * cache_base, const char * stage_dev, size_t group_nbytes) {
    return group_nbytes % sizeof(uint4) == 0 && (uintptr_t) cache_base % sizeof(uint4) == 0 &&
           (uintptr_t) stage_dev % sizeof(uint4) == 0;
}
}  // namespace

// scatter 결과를 되읽어 원본 가중치와 바이트 단위로 비교한다. 실패 수를 누적하고 첫 실패만 보고한다.
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
    // 예산에 들어가는 그룹 수. group_nbytes 가 예산보다 커도 최소 1개는 처리한다.
    const size_t chunk  = std::max<size_t>(1, budget / group_nbytes);

    for (size_t off = 0; off < reload_count; off += chunk) {
        const size_t n = std::min(chunk, reload_count - off);

        g_stage.ensure(n * group_nbytes, (int) n);

        // 1) host 에서 흩어진 그룹을 연속으로 모은다.
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

        // 다음 청크에서 host staging 버퍼를 재사용하므로 완료를 기다린다.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (k_kairox_gather_verify) {
            kairox_gather_verify_chunk(weight_base, cache_base, group_nbytes, stream, reload_plan + off, n);
        }
    }
}


// ---------------------------------------------------------------------------
// zero-copy: GPU 스레드가 pinned host 를 직접 읽어 캐시 슬롯에 쓴다.
//
// PCIe 읽기는 지연이 길고 동시 요청 수로 대역폭을 채운다. 그래서 그룹마다 스레드 하나(행 스레드)로
// 짜면 오히려 느리다 — 워드마다 스레드 하나를 깔아 요청을 최대한 겹친다.
// ---------------------------------------------------------------------------
template <typename T>
static __global__ void kairox_zerocopy_kernel(const T * __restrict__ host_base,
                                              T * __restrict__ cache_base,
                                              const int * __restrict__ group_idx,
                                              const int * __restrict__ slot_idx,
                                              int words_per_group,
                                              int n_groups) {
    size_t       tid   = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) words_per_group * n_groups;
    const size_t step  = (size_t) gridDim.x * blockDim.x;

    for (; tid < total; tid += step) {
        const int g = (int) (tid / words_per_group);
        const int w = (int) (tid % words_per_group);
        cache_base[(size_t) slot_idx[g] * words_per_group + w] =
            host_base[(size_t) group_idx[g] * words_per_group + w];
    }
}

// weight_base 가 디바이스에서 읽을 수 있는 pinned 메모리인지 확인하고, 디바이스 주소를 돌려준다.
// 같은 포인터로 매 호출 물어보지 않도록 마지막 결과를 기억한다 (레이어/텐서마다 base 가 다르므로 작은 캐시).
namespace {
char * kairox_device_ptr_for_host(char * host_ptr) {
    static char * last_host = nullptr;
    static char * last_dev  = nullptr;

    if (host_ptr == last_host) {
        return last_dev;
    }

    cudaPointerAttributes attr = {};
    char *                dev  = nullptr;
    if (cudaPointerGetAttributes(&attr, host_ptr) == cudaSuccess && attr.type == cudaMemoryTypeHost) {
        dev = attr.devicePointer ? (char *) attr.devicePointer : host_ptr;   // UVA 면 같은 주소
    }
    cudaGetLastError();   // 실패 시 남는 에러 플래그를 지운다

    last_host = host_ptr;
    last_dev  = dev;
    return dev;
}
}  // namespace

void kairox_zerocopy_reload(char *              weight_base,
                            char *              cache_base,
                            size_t              group_nbytes,
                            cudaStream_t        stream,
                            const reload_pair * reload_plan,
                            size_t              reload_count) {
    if (reload_count == 0) {
        return;
    }

    char * host_dev_ptr = kairox_device_ptr_for_host(weight_base);
    if (host_dev_ptr == nullptr) {
        // mmap 을 쓰면 pageable 이라 GPU 가 직접 못 읽는다.
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "KAIROX_ZEROCOPY: 가중치가 pinned 가 아니다 (--no-mmap 확인) — gather 경로로 폴백\n");
        }
        kairox_gather_reload(weight_base, cache_base, group_nbytes, stream, reload_plan, reload_count);
        return;
    }

    // 인덱스 전달에만 staging 을 쓴다 (가중치는 복사하지 않는다).
    const size_t chunk = 1024;   // 인덱스 버퍼 크기. 전송 바이트와 무관하다.
    for (size_t off = 0; off < reload_count; off += chunk) {
        const size_t n = std::min(chunk, reload_count - off);

        g_stage.ensure(0, (int) n * 2);
        int * grp_host  = g_stage.slot_host;
        int * slot_host = g_stage.slot_host + n;
        int * grp_dev   = g_stage.slot_dev;
        int * slot_dev  = g_stage.slot_dev + n;

        for (size_t i = 0; i < n; ++i) {
            grp_host[i]  = reload_plan[off + i].group_idx;
            slot_host[i] = reload_plan[off + i].slot_idx;
        }
        CUDA_CHECK(cudaMemcpyAsync(grp_dev, grp_host, 2 * n * sizeof(int), cudaMemcpyHostToDevice, stream));

        const int threads = 1024;   // 워드 스레드. 256 으로 낮추면 요청이 덜 겹쳐 느려진다.
        if (group_nbytes % sizeof(uint4) == 0 && (uintptr_t) cache_base % sizeof(uint4) == 0 &&
            (uintptr_t) host_dev_ptr % sizeof(uint4) == 0) {
            const int    wpg    = (int) (group_nbytes / sizeof(uint4));
            const size_t total  = (size_t) wpg * n;
            const int    blocks = (int) std::min<size_t>(65535, (total + threads - 1) / threads);
            kairox_zerocopy_kernel<uint4><<<blocks, threads, 0, stream>>>(
                (const uint4 *) host_dev_ptr, (uint4 *) cache_base, grp_dev, slot_dev, wpg, (int) n);
        } else {
            const int    wpg    = (int) group_nbytes;
            const size_t total  = (size_t) wpg * n;
            const int    blocks = (int) std::min<size_t>(65535, (total + threads - 1) / threads);
            kairox_zerocopy_kernel<char><<<blocks, threads, 0, stream>>>(
                (const char *) host_dev_ptr, (char *) cache_base, grp_dev, slot_dev, wpg, (int) n);
        }
        CUDA_CHECK(cudaGetLastError());

        // 인덱스 pinned 버퍼를 다음 청크에서 재사용하므로 완료를 기다린다.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (k_kairox_gather_verify) {
            kairox_gather_verify_chunk(weight_base, cache_base, group_nbytes, stream, reload_plan + off, n);
        }
    }
}