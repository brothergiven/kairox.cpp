#include "llama-kairox.h"

#include "ggml-cuda.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <string>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>

static void kairox_encode_ptr(int32_t * op_params, size_t offset, const void * ptr) {
    memcpy(&op_params[offset], &ptr, sizeof(ptr));
}

ggml_tensor * kairox_layer_cache::build_reload_plan(ggml_context * ctx0,
                                                    ggml_tensor *  load_group,
                                                    ggml_tensor *  evict_group) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    kairox_encode_ptr(result->op_params, 0, this);

    result->op     = GGML_OP_RELOAD_PLAN;
    result->src[0] = load_group;
    result->src[1] = evict_group;

    return result;
}

ggml_tensor * kairox_layer_cache::build_reload_exec(ggml_context * ctx0, ggml_tensor * cur, kairox_weight_type wt) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    result->op_params[0] = (int32_t) wt;
    kairox_encode_ptr(result->op_params, 1, this);

    result->op     = GGML_OP_RELOAD_EXEC;
    result->src[0] = cur;

    return result;
}

// Reload 전략을 수행한다. 이 때 가중치는 직접 옮기는게 아니라 메타 정보만을 옮김
// 매 토큰 생성 시, 매 레이어마다 reload plan을 작성한다
void kairox_layer_cache::kairox_reload_plan(const int * load_list, int n_load_in,
                                           const int * evict_list, int n_evict_in) {
    float *   load_group_mask_data   = (float *) load_group_host->data;  // load 할 그룹의 mask 데이터, type: F32, 인덱스는 group 번호, 값은 0.0f 또는 1.0f
    float *   evict_group_mask_data  = (float *) evict_group_host->data; // evict 할 그룹의 mask 데이터
    float *   actual_group_mask_data = (float *) group_mask_host->data;  // 현재 GPU에 올라가있는 Group의 mask 데이터
    int32_t * slot_neuron_idx_data   = (int32_t *) neuron_idx_host->data; // GPU 캐시에 올라가있는 뉴런 번호, 인덱스는 슬롯 번호, 값은 뉴런 번호
    int32_t * slot_of_group_data     = (int32_t *) group_maps->data; // GPU 캐시에 올라가있는 그룹 번호, 인덱스는 그룹 번호, 값은 슬롯 번호
    int32_t * neuron_mask_data       = (int32_t *) neuron_mask->data; // 해당 뉴런이 GPU 캐시에 올라가있는지 여부, 인덱스는 뉴런 번호, 값은 0 또는 1
    const int group_size             = cache_shape.group_size;
    const int n_groups               = cache_shape.n_groups;
    int       n_groups_to_load       = 0;
    int       n_groups_to_evict      = 0;

    // 이미 load, evict 할 그룹은 결정된 상태에서 plan만을 작성한다.
    // 목록은 여기서 만든 순서대로 예산에 잘리므로, 스캔 순서가 곧 우선순위다.
    // KAIROX_RELOAD_ROTATE=1 이면 시작점을 매 스텝 돌려 낮은 번호 편향을 없앤다.
    const auto t_scan0 = k_kairox_profile_plan ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    if (load_list) {
        // GPU 압축 경로: 목록이 이미 와 있으므로 O(n_groups) 순회가 없다.
        // 벡터를 재사용하는 이유는 아래 적용 루프가 groups_to_* 를 읽기 때문이다.
        n_groups_to_load  = std::min(n_load_in, n_groups);
        n_groups_to_evict = std::min(n_evict_in, n_groups);
        std::copy_n(load_list, n_groups_to_load, groups_to_load.begin());
        std::copy_n(evict_list, n_groups_to_evict, groups_to_evict.begin());
    } else {
        const int scan_off = k_kairox_reload_rotate ? (int) (reload_scan_cursor++ % (uint64_t) n_groups) : 0;
        for (int i = 0; i < n_groups; ++i) {
            const int group = scan_off ? (scan_off + i) % n_groups : i;
            if (load_group_mask_data[group]) {
                groups_to_load[n_groups_to_load++] = group;
            }
            if (evict_group_mask_data[group]) {
                // groups_to_evict: layer_cache 구조체 내에 있는 std::vector, 인덱스에 해당하는 그룹을 evict할 예정임을 나타냄
                groups_to_evict[n_groups_to_evict++] = group;
            }
        }
    }
    const auto t_scan1 = k_kairox_profile_plan ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};

    reload_count = 0;
    // top-k 는 |S| = K 를 보장하므로 load 와 evict 개수가 항상 같다. 임계값 선택(KAIROX_NOSORT)은
    // 그 보장이 없어 둘이 어긋난다 — 그때는 비워진 슬롯 수만큼만 교체한다.
    if (!k_kairox_nosort) {
        GGML_ASSERT(n_groups_to_load == n_groups_to_evict); // 로드할 그룹과 evict할 그룹의 개수가 동일해야 함
    }
    reload_planned_count        = n_groups_to_load;
    const int reload_budget     = planned_budget; // reload_budget_groups() : 캐시 대비 비율에 따라 reload할 그룹의 개수를 결정
    const int n_pairs_to_reload = std::min({ n_groups_to_load, n_groups_to_evict, reload_budget });

    for (int i = 0; i < n_pairs_to_reload; ++i) { // 로드할 그룹과 evict할 그룹의 개수만큼 반복
        const int group_to_evict = groups_to_evict[i]; // load, evict 시작
        const int group_to_load  = groups_to_load[i];
        // Eviction 할 그룹의 슬롯 번호를 가져옴, slot_of_group_data: GPU 캐시에 올라가있는 그룹 번호, 인덱스는 그룹 번호, 값은 슬롯 번호
        const int slot           = slot_of_group_data[group_to_evict];

        if (k_kairox_dump_activation) {
            // 스왑 직전 해당 시점에 Evict 될 그룹, Load 될 그룹의 계측 기록.
            // Load 될 그룹에 대해 Total Load + 1 , Use Count 0으로 초기화
            // Evict 될 그룹에 대해 Wasted Load였는지 검사.
            const int evict_base = group_to_evict * group_size; // evict 될 그룹의 시작 뉴런 인덱스
            const int load_base = group_to_load * group_size;   // load 될 그룹의 시작 뉴런 인덱스

            for (int k = 0; k < group_size; k++) {
                // evict 되는 뉴런에 대한 wasted neuron count.
                const int en = evict_base + k; // evicted neuron
                if (dbg_total_loads[en] > 0 && !dbg_used_since_load[en]) {
                    ++dbg_wasted_loads[en]; // 사용되지 않았다면 wasted load
                }

                // load 되는 뉴런에 대한 initialization.
                const int ln = load_base + k; // loaded neuron
                dbg_used_since_load[ln] = 0;
                ++dbg_total_loads[ln];
            }
        }
        // 해당 그룹에 해당하는 뉴런들의 Load 상태를 0으로 만들고, Load 할 그룹에 해당하는 뉴런들의 Load 상태를 1로 만듦
        memset(neuron_mask_data + group_to_evict * group_size, 0, sizeof(int32_t) * group_size);
        std::fill_n(neuron_mask_data + group_to_load * group_size, group_size, 1);

        // base 주소를 계산하여 GPU 캐시에 올라가있는 뉴런들의 인덱스를 업데이트
        const int neuron_base = group_to_load * group_size;
        const int cache_base  = slot * group_size;
        for (int k = 0; k < group_size; ++k) {
            slot_neuron_idx_data[cache_base + k] = neuron_base + k;
        }

        slot_of_group_data[group_to_evict]     = -1;
        slot_of_group_data[group_to_load]      = slot;
        actual_group_mask_data[group_to_evict] = 0.0f;
        actual_group_mask_data[group_to_load]  = 1.0f;

        // 메타데이터 업데이트
        reload_plan[reload_count].group_idx = group_to_load;
        reload_plan[reload_count].slot_idx  = slot;
        ++reload_count;
    }

    // 임계 경로 검증용 지연. 호스트 작업을 늘려 처리량이 그만큼 떨어지는지 본다.
    if (k_kairox_plan_delay_us > 0) {
        const auto t_spin = std::chrono::steady_clock::now();
        const auto want   = std::chrono::microseconds(k_kairox_plan_delay_us);
        while (std::chrono::steady_clock::now() - t_spin < want) {
            // busy wait: sleep 은 스케줄러 지연이 섞여 정밀도가 떨어진다
        }
    }

    if (k_kairox_profile_plan) {
        const auto t_apply1 = std::chrono::steady_clock::now();
        dbg_plan_scan_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_scan1 - t_scan0).count();
        dbg_plan_apply_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_apply1 - t_scan1).count();
        dbg_plan_pairs += (uint64_t) n_pairs_to_reload;
        // |S| 를 알면 임계값이 K 를 맞추고 있는지 볼 수 있다. 캐시가 가득이면 |S| = K + load - evict.
        dbg_plan_load += (uint64_t) n_groups_to_load;
        dbg_plan_evict += (uint64_t) n_groups_to_evict;
        ++dbg_plan_calls;
    }
}

void kairox_init_from_model_and_ctx(struct llama_model *   tgt_model,
                                    struct llama_context * tgt_ctx,
                                    struct llama_model *   dft_model,
                                    struct llama_context * dft_ctx,
                                    const char *           kairox_ms_path,
                                    int64_t                vram_budget) {
    if (kairox_ms_path == nullptr || *kairox_ms_path == '\0') {
        return;
    }

    (void) dft_model;
    size_t free_bytes;
    size_t unused_total_bytes;
    ggml_backend_cuda_get_device_memory(0, &free_bytes, &unused_total_bytes);

    if (vram_budget == 0) {
        vram_budget = free_bytes;
    } else if (vram_budget > 0) {
        const int64_t budget_bytes = vram_budget * 1024 * 1024 * 1024;
        int64_t       used_bytes   = 0;
        for (auto * ctx : { tgt_ctx, dft_ctx }) {
            if (ctx != nullptr) {
                for (auto & buft_size : ctx->memory_breakdown()) {
                    if (ggml_backend_buft_is_host(buft_size.first)) {
                        continue;
                    }
                    used_bytes += buft_size.second.model;
                    used_bytes += buft_size.second.context;
                    used_bytes += buft_size.second.compute;
                }
            }
        }
        vram_budget = std::min<int64_t>(budget_bytes - used_bytes, free_bytes);
    } else {
        GGML_ABORT("fatal error");
    }
    vram_budget -= 512 * (1024 * 1024);  // vram budget margin
    GGML_ASSERT(vram_budget > 0 && "no vram left for initializing cache manager");

    tgt_ctx->kairox_cm = std::make_unique<kairox_cache_manager>(tgt_model, kairox_ms_path, vram_budget);
}

/**
 * kairox_cache_manager의 생성자
 */
kairox_cache_manager::kairox_cache_manager(llama_model * model, const char * kairox_ms_path, int64_t vram_budget) {
    ggml_context *   ctx_meta    = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    gguf_context * ctx_gguf    = gguf_init_from_file(kairox_ms_path, gguf_params);
    const int32_t  n_ffn_group = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_group_size"));
    const float *  ffn_norm_pattern =
        (const float *) gguf_get_arr_data(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_normalized_pattern"));

    ggml_init_params ctx_params = {
        /*.mem_size   = */ ggml_tensor_overhead() * 512,  // magic number here
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ctx_cpu = ggml_init(ctx_params);
    ctx_gpu = ggml_init(ctx_params);

    const auto & layers  = model->layers;
    const int    n_layer = model->hparams.n_layer;
    const int    n_embd  = model->hparams.n_embd;
    const int    n_ff    = model->hparams.n_ff(0);

    layer_caches.resize(n_layer);
    reorder_perms.resize(n_layer);

    auto layer_group_bytes = [&](const llama_layer & layer) {
        size_t bytes = 0;
        bytes += ggml_row_size(layer.ffn_up->type, n_embd) * n_ffn_group;
        if (layer.ffn_gate) {
            bytes += ggml_row_size(layer.ffn_gate->type, n_embd) * n_ffn_group;
        }
        bytes += ggml_row_size(layer.ffn_down_t->type, n_embd) * n_ffn_group;
        return bytes;
    };
    const auto          n_group = n_ff / n_ffn_group;
    std::vector<int>    n_group_cache(n_layer, 0);
    std::vector<size_t> n_bytes_group(n_layer, 0);
    double              n_bytes_group_avg = 0.0;
    for (int il = 0; il < n_layer; ++il) {
        n_bytes_group[il] = layer_group_bytes(layers[il]);
        n_bytes_group_avg += double(n_bytes_group[il]) * ffn_norm_pattern[il];
    }
    const int n_group_cache_budget = std::min<int>(vram_budget / n_bytes_group_avg, n_layer * n_group);
    int       n_group_cache_used   = 0;
    for (int il = 0; il < n_layer; ++il) {
        n_group_cache[il] = std::min<int>(n_group_cache_budget * ffn_norm_pattern[il], n_group);
        n_group_cache_used += n_group_cache[il];
    }
    for (int n_group_cache_left = n_group_cache_budget - n_group_cache_used; n_group_cache_left > 0;) {
        int before_rr = n_group_cache_left;
        for (int il = 0; il < n_layer && n_group_cache_left > 0; ++il) {
            if (n_group_cache[il] < n_group) {
                ++n_group_cache[il];
                --n_group_cache_left;
            }
        }
        if (n_group_cache_left == before_rr) {
            break;
        }
    }
    // // group 의 개수를 1024개 이하로 설정하는 것을 권장함.
    // GGML_ASSERT(n_group <= 1024 && "Recommended: n_group <= 1024 for faster DFR processing");

    auto create_tensor = [&](ggml_context * ctx, ggml_type type, std::vector<int64_t> ne, int il, const char * name) {
        char tensor_name[GGML_MAX_NAME];
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s", il, name);
        ggml_tensor * tensor_meta = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        return ggml_set_name(tensor_meta, tensor_name);
    };

    // il : layer index
    for (int il = 0; il < n_layer; ++il) {
        auto * lc = layer_caches[il] = new kairox_layer_cache(); // layer 마다 layer cache 구조체 생성
        lc->cache_shape              = {
            /*.n_neurons        = */ (int) n_ff,
            /*.n_cached_neurons = */ (int) n_group_cache[il] * n_ffn_group,
            /*.group_size       = */ (int) n_ffn_group,
            /*.n_groups         = */ (int) n_ff / n_ffn_group,
            /*.n_cached_groups  = */ (int) n_group_cache[il],
        };
        lc->reload_plan.resize(lc->cache_shape.n_cached_neurons);
        lc->groups_to_load.resize(lc->cache_shape.n_groups);
        lc->groups_to_evict.resize(lc->cache_shape.n_groups);
        // 되먹임이 켜져 있으면 여기서 시작해 스텝마다 조정되고, 꺼져 있으면 이 값이 고정 상한으로 쓰인다.
        lc->dfr_swap_budget.store(std::clamp(k_kairox_swap_budget_init, k_kairox_swap_budget_min, 1.0f));
        // tau_load 는 lambda 에서 유도된다(Algorithm 1 line 8). ANB 가 켜지면 스텝마다 갱신되고,
        // 꺼져 있으면 0 (= 필터 없음) 또는 KAIROX_TAU_LOAD 로 고정된 값이 그대로 쓰인다.
        lc->dfr_neg_tau = -kairox_tau_load(k_kairox_lambda_init);
        lc->gpu_only = (lc->cache_shape.n_cached_neurons == lc->cache_shape.n_neurons);

        // 계측 플래그 on일 때
        if (k_kairox_dump_activation) {
            lc->dbg_activation_count.assign(lc->cache_shape.n_neurons, 0);
            lc->dbg_resident_count.assign(lc->cache_shape.n_neurons, 0);
            lc->dbg_hit_count.assign(lc->cache_shape.n_neurons, 0);
            lc->dbg_total_loads.assign(lc->cache_shape.n_neurons, 0);
            lc->dbg_wasted_loads.assign(lc->cache_shape.n_neurons, 0);
            lc->dbg_used_since_load.assign(lc->cache_shape.n_neurons, 0);
        }
        // FFN 가중치 값을 layer_cache에 저장
        lc->ffn_pred_up     = layers[il].ffn_pred_up;
        lc->ffn_pred_down   = layers[il].ffn_pred_down;
        lc->ffn_pred_up_b   = layers[il].ffn_pred_up_b;
        lc->ffn_pred_down_b = layers[il].ffn_pred_down_b;

        lc->ffn_up     = layers[il].ffn_up;
        lc->ffn_gate   = layers[il].ffn_gate;
        lc->ffn_down   = layers[il].ffn_down_t;
        lc->ffn_up_b   = layers[il].ffn_up_b;
        lc->ffn_gate_b = layers[il].ffn_gate_b;
        lc->ffn_down_b = layers[il].ffn_down_b;
        lc->ffn_up_cache =
            create_tensor(ctx_gpu, lc->ffn_up->type, { n_embd, lc->cache_shape.n_cached_neurons }, il, "ffn_up.cache");
        if (lc->ffn_gate) {
            lc->ffn_gate_cache = create_tensor(ctx_gpu, lc->ffn_gate->type,
                                               { n_embd, lc->cache_shape.n_cached_neurons }, il, "ffn_gate.cache");
        }
        lc->ffn_down_cache = create_tensor(ctx_gpu, lc->ffn_down->type, { n_embd, lc->cache_shape.n_cached_neurons },
                                           il, "ffn_down.cache");

        lc->neuron_idx =
            create_tensor(ctx_gpu, GGML_TYPE_I32, { lc->cache_shape.n_cached_neurons }, il, "ffn_neuron_idx");
        lc->group_maps  = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->cache_shape.n_groups }, il, "ffn_group_maps");
        lc->neuron_mask = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->cache_shape.n_neurons }, il, "ffn_neuron_mask");
        lc->group_mask  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->cache_shape.n_groups }, il, "ffn_group_mask");
        lc->dfr_scores  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->cache_shape.n_groups }, il, "ffn_dfr_scores");

        lc->neuron_idx_host =
            create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->cache_shape.n_cached_neurons }, il, "ffn_neuron_idx_host");
        lc->load_group_host =
            create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->cache_shape.n_groups }, il, "ffn_load_group_host");
        lc->evict_group_host =
            create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->cache_shape.n_groups }, il, "ffn_evict_group_host");
        lc->group_mask_host =
            create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->cache_shape.n_groups }, il, "ffn_group_mask_host");
        lc->dfr_ema_coeffs = create_tensor(ctx_cpu, GGML_TYPE_F32, { 3 }, il, "ffn_dfr_ema_coeffs");

        reorder_perms[il] =
            create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->cache_shape.n_neurons }, il, "ffn_reorder_perms");
    }
    // group_identity   = create_tensor(ctx_gpu, GGML_TYPE_F32, { n_group, n_group }, 999, "ffn_group_identity");
    sparse_threshold = 0.5f;

    backend_cpu = ggml_backend_cpu_init();
    if (backend_cpu && ggml_get_first_tensor(ctx_cpu)) {
        buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cuda_host_buffer_type());
    }
    backend_gpu = ggml_backend_cuda_init(0);
    if (backend_gpu && ggml_get_first_tensor(ctx_gpu)) {
        buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    }
    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char *  name       = gguf_get_tensor_name(ctx_gguf, i);
        ggml_tensor * src_tensor = ggml_get_tensor(ctx_meta, name);
        ggml_tensor * dst_tensor = ggml_get_tensor(ctx_cpu, name);

        if (src_tensor != nullptr && dst_tensor != nullptr) {
            ggml_backend_tensor_set(dst_tensor, src_tensor->data, 0, ggml_nbytes(dst_tensor));
        }
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    // std::vector<float> f32_mat_buf(n_group * n_group);
    // for (int i = 0; i < n_group; ++i) {
    //     f32_mat_buf[i * n_group + i] = 1.0f;
    // }
    // ggml_backend_tensor_set(group_identity, f32_mat_buf.data(), 0, ggml_nbytes(group_identity));

    std::vector<uint8_t> src_buf_vec(sizeof(float) * n_embd * n_ff);
    std::vector<uint8_t> dst_buf_vec(sizeof(float) * n_embd * n_ff);
    auto *               src_buf = src_buf_vec.data();
    auto *               dst_buf = dst_buf_vec.data();

    auto reorder_tensor_2d = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        const auto n_rows        = ggml_nrows(tensor);
        const auto row_size      = ggml_row_size(tensor->type, tensor->ne[0]);
        const auto row_stride    = tensor->nb[1];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        ggml_backend_tensor_get(tensor, src_buf, 0, tensor_nbytes);
        for (int new_row = 0; new_row < n_rows; ++new_row) {
            const auto old_row = perm[new_row];
            memcpy(dst_buf + new_row * row_stride, src_buf + old_row * row_stride, row_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf, 0, tensor_nbytes);
    };
    auto reorder_tensor_1d = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        const auto n_elem        = tensor->ne[0];
        const auto elem_size     = ggml_row_size(tensor->type, 1);
        const auto elem_stride   = tensor->nb[0];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        ggml_backend_tensor_get(tensor, src_buf, 0, tensor_nbytes);
        for (int new_i = 0; new_i < n_elem; ++new_i) {
            const auto old_i = perm[new_i];
            memcpy(dst_buf + new_i * elem_stride, src_buf + old_i * elem_stride, elem_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf, 0, tensor_nbytes);
    };
    auto reorder_if_exists = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        if (tensor) {
            GGML_ASSERT(ggml_is_contiguous(tensor));
            if (ggml_is_vector(tensor)) {
                reorder_tensor_1d(tensor, perm);
            } else if (ggml_is_matrix(tensor)) {
                reorder_tensor_2d(tensor, perm);
            } else {
                GGML_ABORT("fatal error");
            }
        }
    };

    std::vector<int32_t> perm_vec(n_ff);
    std::vector<int32_t> neuron_idx(n_ff);
    std::vector<int32_t> group_maps(n_ff);
    std::vector<int32_t> neuron_mask(n_ff);
    std::vector<float>   group_mask(n_ff);
    std::vector<float>   dfr_scores(n_ff);
    std::vector<float>   dfr_ema_coeffs({ k_kairox_lambda_init, 1.0f - k_kairox_lambda_init, 1.0f });

    for (int il = 0; il < n_layer; ++il) {
        auto * lc = layer_caches[il];

        auto * reorder_perm = reorder_perms[il];
        ggml_backend_tensor_get(reorder_perm, perm_vec.data(), 0, ggml_nbytes(reorder_perm));
        reorder_if_exists(lc->ffn_up, perm_vec);
        reorder_if_exists(lc->ffn_up_b, perm_vec);
        reorder_if_exists(lc->ffn_gate, perm_vec);
        reorder_if_exists(lc->ffn_gate_b, perm_vec);
        reorder_if_exists(lc->ffn_down, perm_vec);
        reorder_if_exists(lc->ffn_pred_down, perm_vec);
        reorder_if_exists(lc->ffn_pred_down_b, perm_vec);

        size_t cache_nbytes = 0;
        ggml_backend_tensor_set(lc->ffn_up_cache, lc->ffn_up->data, 0, ggml_nbytes(lc->ffn_up_cache));
        cache_nbytes += ggml_nbytes(lc->ffn_up_cache);
        if (lc->ffn_gate) {
            ggml_backend_tensor_set(lc->ffn_gate_cache, lc->ffn_gate->data, 0, ggml_nbytes(lc->ffn_gate_cache));
            cache_nbytes += ggml_nbytes(lc->ffn_gate_cache);
        }
        ggml_backend_tensor_set(lc->ffn_down_cache, lc->ffn_down->data, 0, ggml_nbytes(lc->ffn_down_cache));
        cache_nbytes += ggml_nbytes(lc->ffn_down_cache);

        // [0, 1, ..., m-1]
        std::iota(neuron_idx.begin(), neuron_idx.begin() + lc->cache_shape.n_cached_neurons, 0);
        // [0_0, 1_1, ..., (m/g)-1_(m/g)-1, -1_(m/g), ..., -1_(n/g)-1]
        std::fill_n(group_maps.begin(), lc->cache_shape.n_groups, -1);
        std::iota(group_maps.begin(), group_maps.begin() + lc->cache_shape.n_cached_groups, 0);
        // [1_0, 1_1, ..., 1_m-1, 0_m, ..., 0_n-1]
        std::fill_n(neuron_mask.begin(), lc->cache_shape.n_neurons, 0);
        std::fill_n(neuron_mask.begin(), lc->cache_shape.n_cached_neurons, 1);
        // [1_0, 1_1, ..., 1_(m/g)-1, 0_(m/g), ..., 0_(n/g)-1]
        std::fill_n(group_mask.begin(), lc->cache_shape.n_groups, 0.0f);
        std::fill_n(group_mask.begin(), lc->cache_shape.n_cached_groups, 1.0f);
        // [0.0_0, 0.0_1, ..., 0.0_(m/g)-1, 0_(m/g), ..., 0_(n/g)-1]
        std::fill_n(dfr_scores.begin(), lc->cache_shape.n_groups, 0.0f);

        ggml_backend_tensor_set(lc->neuron_idx, neuron_idx.data(), 0, ggml_nbytes(lc->neuron_idx));
        ggml_backend_tensor_set(lc->neuron_idx_host, neuron_idx.data(), 0, ggml_nbytes(lc->neuron_idx_host));
        ggml_backend_tensor_set(lc->group_maps, group_maps.data(), 0, ggml_nbytes(lc->group_maps));
        ggml_backend_tensor_set(lc->neuron_mask, neuron_mask.data(), 0, ggml_nbytes(lc->neuron_mask));
        ggml_backend_tensor_set(lc->group_mask, group_mask.data(), 0, ggml_nbytes(lc->group_mask));
        ggml_backend_tensor_set(lc->group_mask_host, group_mask.data(), 0, ggml_nbytes(lc->group_mask_host));
        ggml_backend_tensor_set(lc->dfr_scores, dfr_scores.data(), 0, ggml_nbytes(lc->dfr_scores));
        ggml_backend_tensor_set(lc->dfr_ema_coeffs, dfr_ema_coeffs.data(), 0, ggml_nbytes(lc->dfr_ema_coeffs));

        const double cache_n_mega_bytes = cache_nbytes / (1024.0 * 1024.0);
        LLAMA_LOG_INFO("%s: [layer %2d] offloaded %6.2f MiB and cached %5d (%6.2f%%) neurons to device\n", __func__, il,
                       cache_n_mega_bytes, lc->cache_shape.n_cached_neurons,
                       lc->cache_shape.n_cached_neurons * 100.0 / lc->cache_shape.n_neurons);
    }
    LLAMA_LOG_INFO("%s: the cache manager has totally %.2f MiB GPU memory footprint\n", __func__,
                   ggml_backend_buffer_get_size(buf_gpu) / (1024.0 * 1024.0));
}

// 종료 시점에 아직 상주 중인 뉴런은 evict 를 거치지 않아 wasted 판정이 누락된다. 마지막으로 한 번 훑는다.
static void kairox_dump_flush_resident(kairox_layer_cache * lc) {
    const auto * resident = (const int32_t *) lc->neuron_mask->data;
    for (int n = 0; n < lc->cache_shape.n_neurons; ++n) {
        if (resident[n] && lc->dbg_total_loads[n] > 0 && !lc->dbg_used_since_load[n]) {
            lc->dbg_wasted_loads[n] += 1;
        }
    }
}
// Activation 계측 결과를 CSV로 저장
static void kairox_dump_activation_csv(const std::vector<kairox_layer_cache *>  &layer_caches) {
    const char* path_env = getenv("KAIROX_DUMP_ACTIVATION_PATH");
    const std::string path = path_env ? path_env : "kairox_activation.csv";

    FILE *f = fopen(path.c_str(), "w");
    if (!f) {
        LLAMA_LOG_WARN("%s: failed to open '%s' for writing activation dump\n", __func__, path.c_str());
        return;
    }

    fprintf(f, "layer,neuron,activation_count,resident_count,hit_count,total_loads,wasted_loads\n");

    for (size_t il = 0; il < layer_caches.size(); ++il) {
        auto* lc = layer_caches[il];

        kairox_dump_flush_resident(lc);

        for (int n = 0; n < lc->cache_shape.n_neurons; n++) {
            // 뉴런 하나 당 한 줄로 write
            fprintf(f,
                "%zu,%d,%llu,%llu,%llu,%llu,%llu\n",
                il,
                n,
                (unsigned long long) lc->dbg_activation_count[n],
                (unsigned long long) lc->dbg_resident_count[n],
                (unsigned long long) lc->dbg_hit_count[n],
                (unsigned long long) lc->dbg_total_loads[n],
                (unsigned long long) lc->dbg_wasted_loads[n]
            );
        }

    }
    fclose(f);
    LLAMA_LOG_INFO("%s: wrote activation dump to '%s'\n", __func__, path.c_str());
}



// ANB 의 lambda 궤적을 CSV 로 남긴다 (논문 Figure 11 대응). KAIROX_ANB_TRACE=1 일 때만.
static void kairox_dump_anb_trace(const std::vector<kairox_layer_cache *> & layer_caches) {
    const char *      path_env = getenv("KAIROX_ANB_TRACE_PATH");
    const std::string path     = path_env ? path_env : "kairox_anb_trace.csv";

    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        LLAMA_LOG_WARN("%s: failed to open '%s'\n", __func__, path.c_str());
        return;
    }
    fprintf(f, "layer,step,lambda,io_bound\n");
    for (size_t il = 0; il < layer_caches.size(); ++il) {
        const auto * lc = layer_caches[il];
        for (size_t t = 0; t < lc->dbg_anb_lambda.size(); ++t) {
            fprintf(f, "%zu,%zu,%.6f,%d\n", il, t, lc->dbg_anb_lambda[t], (int) lc->dbg_anb_io_bound[t]);
        }
    }
    fclose(f);
    LLAMA_LOG_INFO("%s: wrote ANB trace to '%s'\n", __func__, path.c_str());
}

// RELOAD_PLAN 호스트 구간의 비용을 요약한다 (KAIROX_PROFILE_PLAN).
// 스캔은 O(n_groups) 라 입도를 잘게 하면 커지고, 적용은 O(pairs x group_size) 라 입도와 무관하다.
// 둘 중 어느 쪽이 g=1 의 고정비인지 가르는 것이 목적이다.
static void kairox_dump_plan_profile(const std::vector<kairox_layer_cache *> & layer_caches) {
    uint64_t calls = 0, scan_ns = 0, apply_ns = 0, pairs = 0, n_load = 0, n_evict = 0, cap_sum = 0;
    int      n_groups = 0;
    for (const auto * lc : layer_caches) {
        calls += lc->dbg_plan_calls;
        scan_ns += lc->dbg_plan_scan_ns;
        apply_ns += lc->dbg_plan_apply_ns;
        pairs += lc->dbg_plan_pairs;
        n_load += lc->dbg_plan_load;
        n_evict += lc->dbg_plan_evict;
        cap_sum += (uint64_t) lc->cache_shape.n_cached_groups * lc->dbg_plan_calls;
        n_groups = lc->cache_shape.n_groups;
    }
    if (calls == 0) {
        LLAMA_LOG_INFO("%s: RELOAD_PLAN 호출 0 — KAIROX_PARALLEL 확인\n", __func__);
        return;
    }
    // 레이어 수로 나누면 "토큰 하나를 만드는 동안" 쓴 시간이 된다.
    const uint64_t n_layers = layer_caches.size();
    const uint64_t steps    = n_layers ? calls / n_layers : calls;

    LLAMA_LOG_INFO("%s: === RELOAD_PLAN 호스트 프로파일 (n_groups=%d) ===\n", __func__, n_groups);
    LLAMA_LOG_INFO("%s:   호출        : %llu (%llu 스텝 x %llu 레이어)\n", __func__,
                   (unsigned long long) calls, (unsigned long long) steps, (unsigned long long) n_layers);
    LLAMA_LOG_INFO("%s:   스캔        : %8.3f ms/토큰   (%6.1f ns/호출, %5.2f ns/그룹)\n", __func__,
                   steps ? scan_ns / 1e6 / steps : 0.0, (double) scan_ns / calls,
                   n_groups ? (double) scan_ns / calls / n_groups : 0.0);
    LLAMA_LOG_INFO("%s:   적용        : %8.3f ms/토큰   (%6.1f ns/호출, 짝 %5.1f 개/호출)\n", __func__,
                   steps ? apply_ns / 1e6 / steps : 0.0, (double) apply_ns / calls, (double) pairs / calls);
    LLAMA_LOG_INFO("%s:   합계        : %8.3f ms/토큰\n", __func__,
                   steps ? (scan_ns + apply_ns) / 1e6 / steps : 0.0);
    // |S| = K + load - evict (캐시가 가득일 때). K 대비 비율이 1.0 에서 멀면 임계값이 안 맞는 것이다.
    const double avg_cap = calls ? (double) cap_sum / calls : 0.0;
    const double avg_sel = avg_cap + (double) n_load / calls - (double) n_evict / calls;
    LLAMA_LOG_INFO("%s:   선택 |S|    : %8.1f / K %6.1f  (%5.3f 배)   load %5.1f  evict %5.1f  실행 %5.1f\n",
                   __func__, avg_sel, avg_cap, avg_cap > 0 ? avg_sel / avg_cap : 0.0,
                   (double) n_load / calls, (double) n_evict / calls, (double) pairs / calls);
}

kairox_cache_manager::~kairox_cache_manager() {
    if (k_kairox_dump_activation) {
        kairox_dump_activation_csv(layer_caches);
    }
    if (k_kairox_anb_trace) {
        kairox_dump_anb_trace(layer_caches);
    }
    if (k_kairox_profile_plan) {
        kairox_dump_plan_profile(layer_caches);
    }
    for (auto * const lc : layer_caches) {
        delete lc;
    }

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_free(backend_gpu);

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);
    ggml_backend_free(backend_cpu);
}
