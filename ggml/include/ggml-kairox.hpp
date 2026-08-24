#pragma once

#include "ggml.h"

#include <algorithm>
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
const float k_kairox_lambda_init           = get_env_float("KAIROX_DFR_LAMBDA_INIT", 0.67f);
const float k_kairox_dfr_lambda_adapt_rate = get_env_float("KAIROX_DFR_LAMBDA_ADAPT_RATE", 0.05f);

// 디버그 전용: 뉴런별 "예측 활성화 횟수 / 캐시 상주 횟수"를 누적하고(ggml-cuda.cu의
// ggml_cuda_reload_plan 참고), 프로세스 종료 시 CSV로 덤프하기 위한 전역 On/Off 스위치.
// 기본값 false이므로 평소 벤치마크에는 전혀 영향을 주지 않는다.
const bool  k_kairox_dump_activation        = get_env_bool("KAIROX_DUMP_ACTIVATION", false);

/**
 * @brief FFN 의 weight 를 GPU 로 offload 하고, activation cache 를 GPU 에 유지할 때 사용되는 구조체.
 *        이 구조체 하나가 모델의 레이어 하나에 대응하며, 어떤 뉴런이 GPU에 캐시돼 있는지,
 *        레이어별 predictor 텐서, reload(교체) 진행 상태 등을 전부 들고 있다.
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

    ggml_tensor * load_group_host  = nullptr;
    ggml_tensor * evict_group_host = nullptr;
    ggml_tensor * group_mask_host  = nullptr;
    ggml_tensor * neuron_idx_host  = nullptr;
    ggml_tensor * dfr_ema_coeffs   = nullptr;

    kairox_cache_shape       cache_shape;
    std::vector<reload_pair> reload_plan;
    std::vector<int>         groups_to_load;
    std::vector<int>         groups_to_evict;
    std::atomic<int>         dfr_clamp_k = 0;
    bool                     gpu_only    = false;

    // 디버그 전용 카운터 — KAIROX_DUMP_ACTIVATION=1일 때만 ggml_cuda_reload_plan()에서 채워진다.
    // (시간 가중 지표) 뉴런 n이 "얼마나 자주 필요했는지" / "얼마나 자주 캐시에 있었는지"를 잰다.
    // 배열 크기는 처음 쓸 때 cache_shape.n_neurons로 맞춰짐, 인덱스 n = 이 레이어 안에서의 뉴런 번호.
    std::vector<uint32_t> dbg_activation_count; // [뉴런] 예측기 점수가 임계값 이상이었던 토큰 수 (캐시 여부 무관)
    std::vector<uint32_t> dbg_resident_count;   // [뉴런] 실제로 GPU 캐시에 상주해 있었던 토큰 수
    uint64_t              dbg_token_count = 0;  // 이 레이어에서 위 두 카운터가 관측한 총 토큰 수 (분모)

    // 디버그 전용 — (이벤트 가중 지표) "로드-퇴출 한 사이클" 단위로 낭비 여부를 추적한다.
    // dbg_used_since_load[n]은 뉴런 n이 속한 그룹이 동적으로 reload되어 GPU에 올라오는 순간 0으로
    // 리셋되고, 그 뒤 "GPU에 상주 + 예측기 활성화"로 한 번이라도 관측되면(ggml-cuda.cu의
    // kairox_dump_activation_counts()) 1로 바뀐다. 만약 뉴런의 그룹이 여전히 0인 채로 다시
    // 퇴출(evict)되면, 그 로드는 완전히 낭비된 것이다 — 단 한 토큰도 기여하지 못하고 쫓겨났다는 뜻.
    // 주의: 생성자에서 하는 최초 정적 캐시 채움은 "로드"로 세지 않는다 — 동적 reload 스왑만 센다.
    std::vector<uint8_t>  dbg_used_since_load; // [뉴런] 마지막으로 로드된 이후 한 번이라도 사용됐는가? (0/1)
    std::vector<uint64_t> dbg_total_loads;     // [뉴런] 동적으로 GPU에 로드된 누적 횟수
    std::vector<uint64_t> dbg_wasted_loads;    // [뉴런] 그중 "한 번도 안 쓰이고 그대로 퇴출된" 낭비 횟수

    size_t reload_count         = 0;
    size_t reload_planned_count = 0;
    size_t reload_window_size   = 4;

    kairox_layer_cache()  = default;
    ~kairox_layer_cache() = default;

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

    void make_anchor(KairoxWaitType wait_type, std::atomic<int> * dfr_clamp_k = nullptr, int dfr_clamp_k_cap = 0) {
        AnchorState * anchor = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            GGML_ASSERT(anchor->pending.empty());
            anchor->has_anchor = true;
            anchor->active     = true;
        }

        enqueue_io([this, anchor, dfr_clamp_k, dfr_clamp_k_cap] {
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

            // For simplicity, decrease the maximum load directly when reloading.
            if (k_kairox_dfr_lambda_adapt_rate > 0.0f) {
                if (dfr_clamp_k && dfr_clamp_k_cap > 0) {
                    int cur = dfr_clamp_k->load();
                    int nxt = (int) (cur * (1.0f + (to_move.empty() ? k_kairox_dfr_lambda_adapt_rate :
                                                                      -k_kairox_dfr_lambda_adapt_rate)));
                    dfr_clamp_k->store(std::clamp(nxt, 1, dfr_clamp_k_cap));
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
