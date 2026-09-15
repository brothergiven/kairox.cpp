#pragma once

#include "ggml.h"

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
    ggml_tensor * dfr_ema_coeffs   = nullptr;

    kairox_cache_shape       cache_shape;
    std::vector<reload_pair> reload_plan;
    std::vector<int>         groups_to_load;
    std::vector<int>         groups_to_evict;
    // std::atomic<int>         dfr_clamp_k = 0;
    std::atomic<float>       dfr_swap_budget = { 1.0f }; // 캐시 대비 비율. 1.0이면 캐시 전체를 스왑할 수 있음. 0.5이면 캐시 절반만 스왑 가능
    int                     planned_budget = 0; // RELOAD_PLAN 한 번에 스왑할 수 있는 그룹 개수
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
        return std::clamp((int) std::ceil(dfr_swap_budget.load() * cap), 1, cap); // ceil 로 올림하여 최소 1개 이상, 최대 cap 이하로 제한
    }

    ggml_tensor * build_reload_plan(ggml_context * ctx0, ggml_tensor * load_group, ggml_tensor * evict_group);
    ggml_tensor * build_reload_exec(ggml_context * ctx0, ggml_tensor * cur, kairox_weight_type kairox_wt);
    void          kairox_reload_plan();
};

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

    void make_anchor(KairoxWaitType wait_type, std::atomic<float>* swap_budget = nullptr) {
        AnchorState * anchor = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            GGML_ASSERT(anchor->pending.empty());
            anchor->has_anchor = true;
            anchor->active     = true;
        }

        enqueue_io([this, anchor, swap_budget] {
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
            if (k_kairox_dfr_lambda_adapt_rate > 0.0f && swap_budget) {
                const float cur = swap_budget->load();
                const float nxt = to_move.empty() ? cur * (1.0f + k_kairox_dfr_lambda_adapt_rate)
                                                  : cur / (1.0f + k_kairox_dfr_lambda_adapt_rate);
                swap_budget->store(std::clamp(nxt, k_kairox_swap_budget_min, 1.0f));
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
