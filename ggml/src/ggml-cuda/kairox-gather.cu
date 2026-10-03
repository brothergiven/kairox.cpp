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
// KAIROX_ZEROCOPY_BLOCKS: 동시에 떠 있는 호스트 읽기 요청 수를 줄여 CPU 쪽 DRAM
// 지연을 낮춘다. grid-stride 커널이라 블록을 줄여도 결과는 같다.
static inline int kairox_zerocopy_blocks(size_t want) {
    size_t b = std::min<size_t>(65535, want);
    if (k_kairox_zerocopy_blocks > 0) {
        b = std::min<size_t>(b, (size_t) k_kairox_zerocopy_blocks);
    }
    return (int) std::max<size_t>(1, b);
}

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

/**
 * cudaMemcpyBatchAsync 경로.
 *
 * naive 는 그룹마다 cudaMemcpyAsync 를 부르고 window 마다 동기화한다 — g=1 이면 호출이
 * 16 배가 되어 전송 자체보다 발사 비용이 커진다(실측 25% 손해). gather 는 호출을 1 회로
 * 줄이지만 CPU 가 전송량 전체를 staging 으로 복사한다. zerocopy 는 CPU 복사도 없애지만
 * 커널이 SM 을 점유하는데, g=1 에서는 GPU 가 이미 87% 차 있어 그게 비싸다.
 *
 * 이 경로는 주소 배열 셋만 만들어 한 번에 제출한다. 데이터는 흩어진 채로 복사 엔진이
 * 옮기므로 CPU 복사도 SM 점유도 없다 (nsys 에서 커널이 잡히지 않음을 확인).
 *
 * 실행기가 단일 스레드이므로 주소 배열은 전역 하나를 재사용한다.
 */
void kairox_memcpy_batch_reload(char *              weight_base,
                                char *              cache_base,
                                size_t              group_nbytes,
                                cudaStream_t        stream,
                                const reload_pair * reload_plan,
                                size_t              reload_count) {
    if (reload_count == 0) {
        return;
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    static std::vector<void *>       dsts;
    static std::vector<const void *> srcs;
    static std::vector<size_t>       sizes;

    dsts.resize(reload_count);
    srcs.resize(reload_count);
    sizes.resize(reload_count);

    for (size_t i = 0; i < reload_count; ++i) {
        const reload_pair & p = reload_plan[i];
        dsts[i]  = cache_base  + (size_t) p.slot_idx  * group_nbytes;
        srcs[i]  = weight_base + (size_t) p.group_idx * group_nbytes;
        sizes[i] = group_nbytes;
    }

    // 배치 전체가 스트림 순서로 실행된다. 배치 안의 복사끼리는 순서 보장이 없는데,
    // 슬롯이 서로 달라 의존성이 없으므로 무관하다.
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx     = 0;

    /**
     * 시그니처가 버전마다 다르다 (cudaTypedefs.h 의 PFN_cuMemcpyBatchAsync_v12080 / _v13000):
     *   12.8 ~ 12.x : (..., numAttrs, size_t * failIdx, stream)  — 실패한 복사의 인덱스를 받는다
     *   13.0 ~      : (..., numAttrs,                   stream)  — failIdx 가 빠졌다
     */
    size_t fail_idx = 0;
    (void) fail_idx;
#if CUDART_VERSION >= 13000
    const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                 reload_count, &attr, &attr_idx, 1, stream);
#else
    const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                 reload_count, &attr, &attr_idx, 1, &fail_idx, stream);
#endif
    if (err != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "KAIROX_MEMCPY_BATCH: cudaMemcpyBatchAsync 실패 (%s, fail_idx=%zu) — zerocopy 로 폴백\n",
                    cudaGetErrorString(err), fail_idx);
        }
        kairox_zerocopy_reload(weight_base, cache_base, group_nbytes, stream, reload_plan, reload_count);
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
#else
    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr, "KAIROX_MEMCPY_BATCH: CUDA 12.8 미만으로 빌드됨 — zerocopy 로 폴백\n");
    }
    kairox_zerocopy_reload(weight_base, cache_base, group_nbytes, stream, reload_plan, reload_count);
#endif
}

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
            const int    blocks = kairox_zerocopy_blocks((total + threads - 1) / threads);
            kairox_zerocopy_kernel<uint4><<<blocks, threads, 0, stream>>>(
                (const uint4 *) host_dev_ptr, (uint4 *) cache_base, grp_dev, slot_dev, wpg, (int) n);
        } else {
            const int    wpg    = (int) group_nbytes;
            const size_t total  = (size_t) wpg * n;
            const int    blocks = kairox_zerocopy_blocks((total + threads - 1) / threads);
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
// ============================================================================
// 마스크 → 인덱스 목록 압축 (호스트 스캔 제거)
// ============================================================================

/**
 * 압축 결과를 담는 전역 버퍼. 실행기가 하나뿐이고 태스크가 직렬화되므로 하나를 재사용한다.
 * 레이아웃: [0]=n_load, [1]=n_evict, [2 .. 2+cap)=load, [2+cap .. 2+2*cap)=evict
 */
struct kairox_compact_stage {
    int * dev  = nullptr;  // 디바이스 쪽 버퍼
    int * host = nullptr;  // pinned 호스트 미러
    int   cap  = 0;        // n_groups 상한

    void ensure(int n_groups) {
        if (n_groups <= cap) {
            return;
        }
        if (dev) {
            CUDA_CHECK(cudaFree(dev));
        }
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
        }
        const size_t n_ints = 2 + 2 * (size_t) n_groups;
        CUDA_CHECK(cudaMalloc((void **) &dev, n_ints * sizeof(int)));
        CUDA_CHECK(cudaHostAlloc((void **) &host, n_ints * sizeof(int), cudaHostAllocDefault));
        cap = n_groups;
    }
};

static kairox_compact_stage g_compact;

#define KAIROX_COMPACT_THREADS 1024

/**
 * 마스크가 1 인 인덱스를 모은다. **번호 오름차순을 보존한다.**
 *
 * atomicAdd 로 모으면 순서가 흐트러진다. 그러면 groups_to_load[i] 가
 * slot_of[groups_to_evict[i]] 로 들어갈 때 짝짓기가 뒤섞여, 인접한 그룹이 인접한 슬롯에
 * 놓이던 지역성이 깨진다. 실측에서 그것만으로 교체가 63% 늘었다 (load 289 -> 473).
 *
 * 스레드 하나가 연속 구간을 담당하고 블록 단위 접두합으로 출력 오프셋을 얻는다.
 * 결과는 호스트 스캔과 동일하다. 블록 하나, 커널 한 번, 마스크를 두 번 읽는다(둘째는 L2).
 */
static __global__ void kairox_compact_kernel(const float * __restrict__ load_mask,
                                             const float * __restrict__ evict_mask,
                                             int n_groups,
                                             int * __restrict__ buf) {
    __shared__ int s_l[KAIROX_COMPACT_THREADS];
    __shared__ int s_e[KAIROX_COMPACT_THREADS];

    const int tid   = threadIdx.x;
    const int nthr  = blockDim.x;
    const int chunk = (n_groups + nthr - 1) / nthr;
    const int beg   = tid * chunk;
    const int end   = min(beg + chunk, n_groups);

    // 1 패스: 자기 구간의 개수를 센다
    int cl = 0;
    int ce = 0;
    for (int i = beg; i < end; ++i) {
        cl += (load_mask[i] != 0.0f);
        ce += (evict_mask[i] != 0.0f);
    }
    s_l[tid] = cl;
    s_e[tid] = ce;
    __syncthreads();

    // 2 패스: 블록 단위 포함 접두합 (Hillis-Steele). nthr 은 2 의 거듭제곱이어야 한다.
    for (int off = 1; off < nthr; off <<= 1) {
        const int add_l = (tid >= off) ? s_l[tid - off] : 0;
        const int add_e = (tid >= off) ? s_e[tid - off] : 0;
        __syncthreads();
        s_l[tid] += add_l;
        s_e[tid] += add_e;
        __syncthreads();
    }

    // 포함합에서 자기 개수를 빼면 배타적 오프셋이다
    int pl = s_l[tid] - cl;
    int pe = s_e[tid] - ce;

    if (tid == nthr - 1) {
        buf[0] = s_l[tid];
        buf[1] = s_e[tid];
    }

    // 3 패스: 오프셋부터 번호 순으로 쓴다
    for (int i = beg; i < end; ++i) {
        if (load_mask[i] != 0.0f) {
            buf[2 + pl++] = i;
        }
        if (evict_mask[i] != 0.0f) {
            buf[2 + n_groups + pe++] = i;
        }
    }
}

const int * kairox_compact_masks(const float * load_mask,
                                 const float * evict_mask,
                                 int           n_groups,
                                 cudaStream_t  stream) {
    g_compact.ensure(n_groups);

    // 커널이 카운터를 직접 쓰므로 memset 이 필요 없다. 접두합을 쓰려면 블록 하나여야 한다.
    kairox_compact_kernel<<<1, KAIROX_COMPACT_THREADS, 0, stream>>>(load_mask, evict_mask, n_groups,
                                                                    g_compact.dev);
    CUDA_CHECK(cudaGetLastError());

    // 카운터와 인덱스를 한 번에 내린다. 볼륨은 기존 마스크 두 개와 같은 차수라 문제되지 않는다 —
    // 없애려는 것은 전송량이 아니라 호스트의 O(n_groups) 순회다.
    CUDA_CHECK(cudaMemcpyAsync(g_compact.host, g_compact.dev, (2 + 2 * (size_t) n_groups) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return g_compact.host;
}
