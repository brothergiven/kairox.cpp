#pragma once

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <vector>

enum kairox_weight_type { KAIROX_FFN_UP = 1, KAIROX_FFN_GATE, KAIROX_FFN_DOWN };

typedef struct {
    int n_neurons;
    int n_cached_neurons;
    int group_size;
    int n_groups;
    int n_cached_groups;
} kairox_cache_shape;

typedef struct {
    int group_idx, slot_idx;
} reload_pair;

inline float get_env_float(const char * env, float default_value) {
    if (const char * p = getenv(env)) {
        char * end = nullptr;
        float  v   = strtof(p, &end);
        if (end != p && *end == '\0') {
            return v;
        }
    }
    return default_value;
}

inline int get_env_int(const char * env, int default_value) {
    if (const char * p = getenv(env)) {
        char * end = nullptr;
        long   v   = strtol(p, &end, 10);
        if (end != p && *end == '\0') {
            return (int) v;
        }
    }
    return default_value;
}

inline bool get_env_bool(const char * env, bool default_value) {
    if (const char * p = getenv(env)) {
        char * end = nullptr;
        long   v   = strtol(p, &end, 10);
        if (end != p && *end == '\0' && (v == 0 || v == 1)) {
            return v == 1;
        }
    }
    return default_value;
}

const bool  k_enable_kairox_parallel       = get_env_bool("KAIROX_PARALLEL", false);
const float k_kairox_lambda_init           = get_env_float("KAIROX_DFR_LAMBDA_INIT", 0.67f); // lambda 초기 값
const float k_kairox_dfr_lambda_adapt_rate = get_env_float("KAIROX_DFR_LAMBDA_ADAPT_RATE", 0.05f);
const float k_kairox_swap_budget_min      = get_env_float("KAIROX_SWAP_BUDGET_MIN", 0.01f); // swap budget 최소값
/**
 * 저자 배포 구현의 원본 예산 제어로 되돌린다 (절제 실험용 기준선).
 *
 * 배포본은 그룹 *개수* 를 정수로 들고 병목 신호에 따라 `(int)(cur * (1 +- alpha))` 로 갱신했다.
 * 결함이 둘이다.
 *   (a) 1.05f 는 실제로 1.04999995 라 올라가려면 cur >= 21 이어야 한다. cur <= 20 에서는
 *       (int) 절단에 먹혀 증가가 불가능하고, 감소만 된다. 1 은 흡수 상태다
 *       ((int)(1 * 1.05) == 1). 한 번 20 아래로 떨어지면 영구히 1 로 수렴한다.
 *   (b) x(1+a) 와 x(1-a) 를 번갈아 하면 곱이 1-a^2 < 1 이라 20 위에서도 단조 하강한다.
 *       n_cached_groups 에서 1 까지 log(n)/log(1/(1-a)) 스텝 — opt-6.7b 는 약 124 스텝.
 * 그래서 100~200 토큰 안에 레이어-스텝당 1 그룹으로 얼어붙고, 동적 분할이 사실상 꺼진다.
 *
 * 수정판(기본)은 캐시 대비 실수 비율로 들고 x(1+a) / /(1+a) 로 대칭 갱신한다 —
 * 갇힘이 없고 g 와 무관하게 같은 뉴런 수를 교체한다.
 *
 * 이 손잡이를 켜면 원본 경로가 돌아온다. "우리가 무엇을 이겼는가" 를 말하려면
 * 수정판이 아니라 이쪽이 기준선이어야 한다.
 */
const bool k_kairox_clamp_int = get_env_bool("KAIROX_CLAMP_INT", false);
/**
 * 스왑 예산의 초기값. 되먹임을 끄면(KAIROX_DFR_LAMBDA_ADAPT_RATE=0) 이 값이 고정 상한으로 계속 쓰인다.
 * 캐시 대비 비율이라 group_size 와 무관하다 — ratio x n_cached_groups x group_size = ratio x n_cached_neurons.
 * 즉 g 를 바꿔도 같은 비율이면 같은 바이트가 움직이므로, 입도 비교를 전송량 고정으로 할 수 있다.
 */
const float k_kairox_swap_budget_init     = get_env_float("KAIROX_SWAP_BUDGET", 1.0f);
/**
 * 1 이면 예산으로 잘릴 때 스캔 시작점을 매 스텝 돌린다.
 * 지금 plan 은 그룹 번호 오름차순으로 만들어져 앞에서부터 잘리므로, 예산을 조이면
 * 낮은 번호 그룹만 체계적으로 살아남는다. 라운드로빈으로 그 편향을 없앤다.
 */
const bool k_kairox_reload_rotate          = get_env_bool("KAIROX_RELOAD_ROTATE", false);
/**
 * RELOAD_PLAN 의 호스트 구간을 스캔/적용으로 나눠 잰다.
 *   스캔 : 마스크 전체를 훑어 번호 목록을 만든다     — O(n_groups),      g 에 반비례해 커진다
 *   적용 : 짝지어 메타데이터를 갱신한다               — O(pairs x g),     g 와 무관하다
 * 입도를 잘게 할 때 늘어나는 고정비가 어디에 있는지 가르기 위한 것이다.
 */
const bool k_kairox_profile_plan           = get_env_bool("KAIROX_PROFILE_PLAN", false);
/**
 * CPU 팔 프로파일 (KAIROX_PROFILE_CPU).
 *
 * CPU 희소 연산은 kairox_executor 워커에 비동기로 제출되고 나중에 split_fut.get() 으로
 * 합류한다. 그래서 세 지점을 재면 두 팔의 선후가 그대로 나온다.
 *
 *   join    메인이 CPU 팔을 기다린 시간      -> CPU 가 임계 팔일 때의 *노출된* 비용
 *   evsync  CPU split 이 GPU 이벤트를 기다린 시간 -> GPU 가 앞서 있다는 뜻
 *   work    워커가 실제로 계산한 시간         -> CPU 팔의 크기 자체
 *
 * 셋 다 블로킹 호출 앞뒤의 *경과* 시간이다. ggml 풀이 스핀 대기해도 부풀지 않는다 —
 * perf 로 CPU 사용률을 재면 스핀이 섞여 못 쓴다. 그래서 지금까지 CPU 를 못 쟀다.
 *
 *   join ~ 0      CPU 가 필요해지기 전에 끝났다. 다른 팔에 가려져 있다
 *   join 이 크다  CPU 가 임계 팔이고 그 값이 곧 비용이다
 */
const bool k_kairox_profile_cpu = get_env_bool("KAIROX_PROFILE_CPU", false);

struct kairox_cpu_profile {
    std::atomic<uint64_t> join_ns{ 0 },   join_cnt{ 0 };
    std::atomic<uint64_t> evsync_ns{ 0 }, evsync_cnt{ 0 };
    std::atomic<uint64_t> work_ns{ 0 },   work_cnt{ 0 };
    std::atomic<uint64_t> steps{ 0 };     // 그래프 실행 횟수 = 디코드 토큰 수
};
inline kairox_cpu_profile g_kairox_cpu_prof;

inline uint64_t kairox_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 블로킹 구간을 재는 작은 스코프 가드. KAIROX_PROFILE_CPU 가 꺼져 있으면 아무것도 안 한다.
struct kairox_span {
    std::atomic<uint64_t> * ns;
    std::atomic<uint64_t> * cnt;
    uint64_t                t0;
    kairox_span(std::atomic<uint64_t> & a, std::atomic<uint64_t> & c)
        : ns(k_kairox_profile_cpu ? &a : nullptr), cnt(k_kairox_profile_cpu ? &c : nullptr),
          t0(k_kairox_profile_cpu ? kairox_now_ns() : 0) {}
    ~kairox_span() {
        if (ns) {
            ns->fetch_add(kairox_now_ns() - t0, std::memory_order_relaxed);
            cnt->fetch_add(1, std::memory_order_relaxed);
        }
    }
};
/**
 * 절제 실험용. argsort_top_k + index_mask 대신 고정 임계값 비교로 S 를 만든다.
 * topk_idx 는 마스크를 만드는 데만 쓰이고 순서는 버려지므로, 정렬은 원래 필요하지 않다.
 * 임계값을 제어하지 않으면 |S| 가 K 와 어긋나 정책 품질은 나빠진다 — 비용만 재는 용도다.
 * 임계값은 KAIROX_TAU_LOAD 로 준다.
 */
const bool k_kairox_nosort                 = get_env_bool("KAIROX_NOSORT", false);
/**
 * RELOAD_PLAN 의 호스트 구간에 지연을 주입한다 (마이크로초, 호출당).
 * 의미는 전혀 바꾸지 않는다 — 호스트 작업이 임계 경로에 있는지 재기 위한 것이다.
 * 주입한 시간만큼 처리량이 그대로 떨어지면(기울기 1) 완전히 노출된 것이고,
 * 덜 떨어지면 GPU 작업과 겹쳐 일부가 가려진다는 뜻이다.
 */
const int  k_kairox_plan_delay_us          = get_env_int("KAIROX_PLAN_DELAY_US", 0);
/**
 * 마스크 → 인덱스 목록 압축을 GPU 에서 한다. 호스트의 O(n_groups) 스캔이 O(pairs) 가 된다.
 * 계측(KAIROX_DUMP_ACTIVATION)은 호스트 적용 루프 안에 있으므로 이 경로에서도 그대로 동작한다.
 */
const bool k_kairox_gpu_compact            = get_env_bool("KAIROX_GPU_COMPACT", false);
const bool k_kairox_dump_activation        = get_env_bool("KAIROX_DUMP_ACTIVATION", false); // 환경변수로 activation 계측 할 건지 결정.
// --- 결정 단위 / 전송 단위 분리 (gather & scatter) ---------------------------
// 원본 reload 는 그룹 하나마다 cudaMemcpyAsync 를 한 번씩 호출해 전송 시간이 바이트가 아니라 호출 수에 묶인다.
// 흩어진 그룹을 pinned staging 버퍼로 모아 H2D 1회 + GPU scatter 커널 1회로 옮긴다.
// 기본 false — 원본 경로를 그대로 두고 env 로만 켠다.
const bool k_kairox_gather            = get_env_bool("KAIROX_GATHER", false);
// staging 버퍼 바이트 예산(MiB). 한 번의 reload 가 이보다 크면 청크로 나눈다.
const int  k_kairox_gather_budget_mib = get_env_int("KAIROX_GATHER_BUDGET_MIB", 64);
// scatter 직후 GPU 캐시 슬롯을 되읽어 원본 가중치와 바이트 비교한다. 매우 느리다 — 정확성 검증 전용.
// (KAIROX 는 reload 가 compute 와 비동기로 경쟁해 같은 seed 로도 출력이 달라지므로 출력 비교로는 검증 불가)
const bool k_kairox_gather_verify     = get_env_bool("KAIROX_GATHER_VERIFY", false);
// GPU 가 pinned host 가중치를 직접 읽어 캐시 슬롯에 쓰는 경로 (zero-copy).
// host staging gather 와 달리 CPU memcpy 와 staging 버퍼가 없다 — 옮기는 PCIe 바이트는 같고,
// DDR 왕복 한 번과 VRAM 내부 복사 한 번이 사라진다. 대신 PCIe 응답을 기다리는 동안 SM 을 점유한다.
// --no-mmap 이라 CPU 가중치가 pinned 버퍼에 있어야 동작한다. 아니면 gather 경로로 폴백한다.
const bool k_kairox_zerocopy = get_env_bool("KAIROX_ZEROCOPY", false);

/**
 * cudaMemcpyBatchAsync (CUDA 12.8+) 경로. 흩어진 복사 n 개를 API 호출 1 회로 제출한다.
 * gather 처럼 데이터를 모으지 않고 "명령"만 모으므로 CPU 복사가 없고, zerocopy 와 달리
 * 커널을 띄우지 않아 SM 을 점유하지 않는다 (복사 엔진이 처리한다).
 * g=1 에서 naive 는 호출이 16 배가 되고 zerocopy 는 SM 을 먹는데, 이 경로는 둘 다 피한다.
 */
const bool k_kairox_memcpy_batch = get_env_bool("KAIROX_MEMCPY_BATCH", false);

/**
 * Adaptive Neuron Balancer (논문 Algorithm 1 Phase 1) 설정.
 *
 * KAIROX_ANB=1 이면 병목 피드백이 lambda 를 조절한다(논문 동작). tau_load 도 lambda 를 따라간다.
 * KAIROX_ANB=0 (기본) 이면 lambda 를 고정하고, 대신 스왑 예산(dfr_swap_budget)을 조절한다(배포 구현 계열).
 * lambda 의 상/하한은 논문에 수치가 없다. Figure 11 에서 lambda 가 0.85 부근까지 올라가므로 그 위로 여유를 뒀다.
 */
const bool  k_enable_kairox_anb = get_env_bool("KAIROX_ANB", false);
const float k_kairox_lambda_min = get_env_float("KAIROX_DFR_LAMBDA_MIN", 0.10f);
const float k_kairox_lambda_max = get_env_float("KAIROX_DFR_LAMBDA_MAX", 0.95f);
// tau_load = (1 - lambda) + eps 의 판별 마진
const float k_kairox_tau_eps = get_env_float("KAIROX_TAU_EPS", 1e-6f);
// 0 보다 크면 tau_load 를 lambda 와 무관하게 이 값으로 고정한다 (필터 강도 스윕용)
const float k_kairox_tau_load_fixed = get_env_float("KAIROX_TAU_LOAD", 0.0f);
// 레이어별 lambda 궤적을 기록해 종료 시 CSV 로 덤프 (논문 Figure 11 대응)
const bool k_kairox_anb_trace = get_env_bool("KAIROX_ANB_TRACE", false);

// 논문 Algorithm 1 line 8: tau_load <- (1 - lambda) + eps
inline float kairox_tau_load(float lambda) {
    if (k_kairox_tau_load_fixed > 0.0f) {
        return k_kairox_tau_load_fixed;
    }
    // ANB 를 끈 기본 상태에서는 tau 필터를 쓰지 않는다 (이 저장소의 현재 동작 유지).
    if (!k_enable_kairox_anb) {
        return 0.0f;
    }
    /**
     * lambda = 0 은 관성이 아예 없는 경우다(neuralink 프로파일). 이때 S = A <= 1 인데
     * (1 - lambda) + eps = 1 + eps 라 어떤 그룹도 필터를 통과하지 못해 캐시가 정적으로 굳는다.
     * 매 스텝이 독립적인 결정이라 one-hit wonder 라는 개념 자체가 없으므로 필터를 끈다.
     */
    if (lambda <= 0.0f) {
        return 0.0f;
    }
    return (1.0f - lambda) + k_kairox_tau_eps;
}
/**
 * 캐시 관리 정책을 실제로 수행하는 구조체
 * 각 레이어마다 하나씩 존재하며 , 레이어의 FFN 가중치를 메모리로 로드하고
 */
struct kairox_layer_cache {
    ggml_tensor * ffn_pred_up     = nullptr;
    ggml_tensor * ffn_pred_down   = nullptr;
    ggml_tensor * ffn_pred_up_b   = nullptr;
    ggml_tensor * ffn_pred_down_b = nullptr;
    ggml_tensor * ffn_up          = nullptr;
    ggml_tensor * ffn_gate        = nullptr;
    ggml_tensor * ffn_down        = nullptr;
    ggml_tensor * ffn_up_b        = nullptr;
    ggml_tensor * ffn_gate_b      = nullptr;
    ggml_tensor * ffn_down_b      = nullptr;
    ggml_tensor * ffn_up_cache    = nullptr;
    ggml_tensor * ffn_gate_cache  = nullptr;
    ggml_tensor * ffn_down_cache  = nullptr;

    ggml_tensor * neuron_idx  = nullptr;
    ggml_tensor * group_maps  = nullptr;
    ggml_tensor * neuron_mask = nullptr;
    ggml_tensor * group_mask  = nullptr;
    ggml_tensor * dfr_scores  = nullptr;

    ggml_tensor * sparse_idx  = nullptr;
    ggml_tensor * reload_up   = nullptr;
    ggml_tensor * reload_gate = nullptr;
    ggml_tensor * reload_down = nullptr;

    // Host-side tensors for managing reload plans and execution
    // 전부 인덱스: 그룹 번호, 값 = 0.0f 또는 1.0f 인 비트
    ggml_tensor * load_group_host  = nullptr;
    ggml_tensor * evict_group_host = nullptr;
    ggml_tensor * group_mask_host  = nullptr; // 현재 GPU에 올라간 그룹을 나타내는 비트 벡터, 교체가 발생하면 0->1, 1->0으로 바뀜
    // mask 형태를 사용하는 이유가 Sparse Matrix Multiply를 그대로 사용하기 위해서임, 즉, Sparse Matrix Multiply를 수행할 때, mask가 1인 그룹만을 사용하여 Multiply를 수행함
    ggml_tensor * neuron_idx_host  = nullptr;
    ggml_tensor * dfr_ema_coeffs   = nullptr; // {lambda, 1 - lambda, normalizer}. ANB 가 앞의 두 값을 매 스텝 갱신한다

    /**
     * -tau_load. ggml_shifted_step_dyn 이 실행 시점에 이 주소를 읽는다.
     * 커널이 (x + threshold) > 0 을 계산하므로 부호를 뒤집어 저장한다 -- 즉 score > tau_load.
     * lambda 가 ANB 로 바뀔 때마다 같이 갱신되며, 그래프가 재사용돼도 최신 값이 반영된다.
     */
    float dfr_neg_tau = 0.0f;

    // ANB 궤적 (KAIROX_ANB_TRACE). executor 스레드에서만 갱신하고 종료 시 덤프한다.
    std::vector<float>   dbg_anb_lambda;
    std::vector<uint8_t> dbg_anb_io_bound;

    kairox_cache_shape       cache_shape;
    std::vector<reload_pair> reload_plan;
    std::vector<int>         groups_to_load;
    std::vector<int>         groups_to_evict;
    // KAIROX_CLAMP_INT: 저자 배포본의 정수 그룹 예산. init 에서 n_cached_groups 로 채운다.
    std::atomic<int>         dfr_clamp_k = { 0 };
    std::atomic<float>       dfr_swap_budget = { 1.0f }; // 캐시 대비 비율. 1.0이면 캐시 전체를 스왑할 수 있음. 0.5이면 캐시 절반만 스왑 가능
    int                     planned_budget = 0; // RELOAD_PLAN 한 번에 스왑할 수 있는 그룹 개수
    uint64_t                reload_scan_cursor = 0; // KAIROX_RELOAD_ROTATE: plan 스캔 시작점. executor 스레드에서만 쓴다
    // KAIROX_PROFILE_PLAN. executor 스레드에서만 갱신하고 종료 시 합산한다
    uint64_t                dbg_plan_calls  = 0;
    uint64_t                dbg_plan_scan_ns = 0;
    uint64_t                dbg_plan_apply_ns = 0;
    uint64_t                dbg_plan_pairs  = 0;
    uint64_t                dbg_plan_load   = 0;
    uint64_t                dbg_plan_evict  = 0;
    std::vector<float>      dfr_score_host;
    bool                     gpu_only    = false;

    /**
     * 뉴런별 디버그 전용 배열. 크기 : n_neurons
     * 1. Activation Count : 해당 Layer에서 Activation 된 횟수, INT64
     * 2. Resident Count : 해당 Layer에서 GPU에 올라간 뉴런의 개수, INT64
     * 3. Use Count : 해당 Layer에서 GPU에 올라간 뉴런이 실제로 사용된 횟수, INT64
     * 4. Total Loads : 해당 뉴런이 Reload 된 횟수, INT64
     * 5. Wasted Loads : 해당 뉴런이 Reload 되었지만 실제로 사용되지 않은 횟수, INT64
     * 6. Used Since Load: 해당 뉴런이 Load 되고 나서 사용되었는지 플래그, INT8
     *
     * 최종 얻고자 하는 데이터
     * 1. Layer별 (Resident && Active) / (Resident), 즉 상주하던 뉴런 중 활성화 된 counts
     * 2. Layer별 (Resident && Active) / (Active), 즉 활성화된 뉴런 중 상주 하던 counts
     * 3. Wasted Reload : (Wasted) / (Total)
     */
    std::vector<uint64_t> dbg_activation_count;
    std::vector<uint64_t> dbg_resident_count;
    std::vector<uint64_t> dbg_hit_count;
    std::vector<uint64_t> dbg_total_loads;
    std::vector<uint64_t> dbg_wasted_loads;
    std::vector<uint8_t> dbg_used_since_load; //
    size_t reload_count         = 0; // Reload Count가 구조체 내에 존재한다(해당 Layer가 Reload Plan을 수행한 횟수)
    size_t reload_planned_count = 0;
    size_t reload_window_size   = 4;

    kairox_layer_cache()  = default;
    ~kairox_layer_cache() = default;

    int reload_budget_groups() const {
        const int cap = cache_shape.n_cached_groups;
        if (k_kairox_clamp_int) {
            return std::clamp(dfr_clamp_k.load(), 1, cap); // 저자 배포본 경로
        }
        return std::clamp((int) std::ceil(dfr_swap_budget.load() * cap), 1, cap); // ceil 로 올림하여 최소 1개 이상, 최대 cap 이하로 제한
    }

    ggml_tensor * build_reload_plan(ggml_context * ctx0, ggml_tensor * load_group, ggml_tensor * evict_group);
    ggml_tensor * build_reload_exec(ggml_context * ctx0, ggml_tensor * cur, kairox_weight_type kairox_wt);
    /**
     * reload plan 을 작성한다.
     * 목록을 주면(GPU 압축 경로) 호스트가 마스크를 훑지 않고 그 목록을 그대로 쓴다.
     * 주지 않으면 기존대로 load/evict 마스크를 n_groups 만큼 스캔한다.
     */
    void          kairox_reload_plan(const int * load_list  = nullptr, int n_load_in  = 0,
                                     const int * evict_list = nullptr, int n_evict_in = 0);
};

/**
 * 논문 Algorithm 1, Phase 1: Adaptive Balancing-Intensity Control.
 *
 *   1: Feedback <- GetSystemBottleneck()
 *   2: if IO_BOUND   then lambda <- min(lambda * (1 + alpha), lambda_max)
 *   4: elif CPU_BOUND then lambda <- max(lambda * (1 - alpha), lambda_min)
 *
 * lambda 를 올리면 관성이 커져 차가운 그룹이 tau_load 를 넘기 어려워진다(= 보수적, I/O 감소).
 * 내리면 최근 활성화에 민감해져 교체가 늘고 CPU 계산이 줄어든다.
 * 배포 구현이 스왑 예산을 직접 자르는 것과 달리, 이쪽은 "무엇을 올릴지" 자체를 바꾼다.
 */
inline void kairox_anb_feedback(kairox_layer_cache * lc, bool io_bound) {
    auto * coeffs = (float *) lc->dfr_ema_coeffs->data;  // {lambda, 1 - lambda, normalizer}

    const float alpha  = k_kairox_dfr_lambda_adapt_rate;
    const float lambda = io_bound ? std::min(coeffs[0] * (1.0f + alpha), k_kairox_lambda_max) :
                                    std::max(coeffs[0] * (1.0f - alpha), k_kairox_lambda_min);

    // TAM 업데이트(ggml_scale_add)가 이 배열을 실행 시점에 읽는다. 한 스텝 어긋나도 점수가
    // 조금 흔들릴 뿐이라 락은 걸지 않는다.
    coeffs[1] = 1.0f - lambda;
    coeffs[0] = lambda;

    lc->dfr_neg_tau = -kairox_tau_load(lambda);

    if (k_kairox_anb_trace) {
        lc->dbg_anb_lambda.push_back(lambda);
        lc->dbg_anb_io_bound.push_back(io_bound ? 1 : 0);
    }
}

void ggml_cuda_set_device(int device);

// kairox async kernel caller and io executor
struct SingleThreadExecutor {
    enum KairoxWaitType { KAIROX_WAIT_MUL_MAT_SPARSE = 0, KAIROX_WAIT_AXPY_SPARSE };

    SingleThreadExecutor() {
        worker_ = std::thread([this] {
            ggml_cuda_set_device(0);
            loop();
        });
    }

    SingleThreadExecutor(const SingleThreadExecutor &)             = delete;
    SingleThreadExecutor & operator=(const SingleThreadExecutor &) = delete;
    SingleThreadExecutor(SingleThreadExecutor &&)                  = delete;
    SingleThreadExecutor & operator=(SingleThreadExecutor &&)      = delete;

    ~SingleThreadExecutor() { stop(); }

    template <class F, class... Args> static auto make_bound(F && f, Args &&... args) {
        using Fn  = std::decay_t<F>;
        using Tup = std::tuple<std::decay_t<Args>...>;

        return [fn = Fn(std::forward<F>(f)), tup = Tup(std::forward<Args>(args)...)]() mutable {
            return std::apply(fn, tup);
        };
    }

    template <class F, class... Args> void post(F && f, Args &&... args) {
        auto bound = make_bound(std::forward<F>(f), std::forward<Args>(args)...);
        enqueue_io(std::move(bound));
    }

    template <class F, class... Args> auto submit(KairoxWaitType wait_type, F && f, Args &&... args) {
        using R = std::invoke_result_t<F, Args...>;

        auto bound    = make_bound(std::forward<F>(f), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::move(bound));
        auto fut      = task_ptr->get_future();
        auto wrapper  = [task_ptr]() {
            (*task_ptr)();
        };

        bool          need_notify = false;
        AnchorState * anchor      = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);

            if (!anchor->has_anchor || !anchor->active) {
                tasks_.emplace_back(std::move(wrapper));
                need_notify = true;
            } else {
                anchor->pending.emplace_back(std::move(wrapper));
            }
        }

        if (need_notify) {
            cv_.notify_one();
        }

        return fut;
    }

    // lc 를 넘기면 anchor 가 풀리는 시점에 병목을 판정해 balancing 강도를 조절한다.
    //   KAIROX_ANB=1 : lambda 를 조절한다 (논문 Algorithm 1 Phase 1)
    //   KAIROX_ANB=0 : 스왑 예산 비율을 조절한다 (배포 구현 계열, 기본)
    void make_anchor(KairoxWaitType wait_type, kairox_layer_cache * lc = nullptr) {
        AnchorState * anchor = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            GGML_ASSERT(anchor->pending.empty());
            anchor->has_anchor = true;
            anchor->active     = true;
        }

        enqueue_io([this, anchor, lc] {
            std::deque<std::function<void()>> to_move;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                to_move.swap(anchor->pending);
                anchor->active = false;

                for (auto & fn : to_move) {
                    tasks_.emplace_back(std::move(fn));
                }
            }

            if (!to_move.empty()) {
                cv_.notify_one();
            }

            // swap budget을 조정하는 로직을 여기에 추가할 수 있음
            // swap budget 이 조정되는 규칙은
            // 1. to_move 가 비어있으면 swap budget을 증가시킴 (더 많은 그룹을 스왑할 수 있도록)
            // 2. to_move 가 비어있지 않으면 swap budget을 감소시킴
            // 3. 현재 swap budget 값에 k_kairox_dfr_lambda_adapt_rate를 곱하여 조정
            //
            // anchor 에 매달린 task 가 있었다면(= to_move 비어있지 않음) 계산이 전송을 기다린 것이므로 IO 병목,
            // 비어 있었다면 전송이 먼저 끝났으므로 CPU 병목으로 본다 (Algorithm 1 line 1, GetSystemBottleneck).
            if (k_kairox_dfr_lambda_adapt_rate > 0.0f && lc) {
                const bool io_bound = !to_move.empty();
                if (k_enable_kairox_anb) {
                    kairox_anb_feedback(lc, io_bound);
                } else if (k_kairox_clamp_int) {
                    // 저자 배포본 그대로. (int) 절단과 1-a^2 하강을 일부러 보존한다 —
                    // 이 경로의 결함이 측정 대상이다.
                    const int cur = lc->dfr_clamp_k.load();
                    const int nxt = (int) (cur * (1.0f + (io_bound ? -k_kairox_dfr_lambda_adapt_rate
                                                                   : k_kairox_dfr_lambda_adapt_rate)));
                    lc->dfr_clamp_k.store(std::clamp(nxt, 1, lc->cache_shape.n_cached_groups));
                } else {
                    const float cur = lc->dfr_swap_budget.load();
                    const float nxt = io_bound ? cur / (1.0f + k_kairox_dfr_lambda_adapt_rate)
                                               : cur * (1.0f + k_kairox_dfr_lambda_adapt_rate);
                    lc->dfr_swap_budget.store(std::clamp(nxt, k_kairox_swap_budget_min, 1.0f));
                }
            }
        });
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!worker_.joinable()) {
                return;
            }
            tasks_.emplace_back(std::function<void()>{});
        }
        cv_.notify_one();
        worker_.join();
    }

    struct AnchorState {
        bool                              has_anchor = false;
        bool                              active     = false;
        std::deque<std::function<void()>> pending;
    };

    AnchorState anchor_mm_sparse_;
    AnchorState anchor_axpy_sparse_;

    AnchorState * anchor_ref(KairoxWaitType wait_type) {
        switch (wait_type) {
            case KAIROX_WAIT_MUL_MAT_SPARSE:
                return &anchor_mm_sparse_;
            case KAIROX_WAIT_AXPY_SPARSE:
                return &anchor_axpy_sparse_;
            default:
                GGML_ABORT("anchor_ref: invalid wait_type");
        }
    }

    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> tasks_;
    std::deque<std::function<void()>> io_tasks_;
    std::thread                       worker_;

    template <class F> void enqueue_io(F && fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            io_tasks_.emplace_back(std::forward<F>(fn));
        }
        cv_.notify_one();
    }

    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return !tasks_.empty() || !io_tasks_.empty(); });

                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                } else if (!io_tasks_.empty()) {
                    task = std::move(io_tasks_.front());
                    io_tasks_.pop_front();
                } else {
                    continue;
                }
            }

            if (!task) {
                break;
            }

            task();
        }
    }
};
