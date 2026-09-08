# 디코드 스텝 시간 분해 계측 (KAIROX_PROFILE)

브랜치: `exp/step-profiling` (base `exp/group-granularity`)

## 1. 왜 만들었나

group_size 스윕(`bench_group_sweep.sh`, 7조합 x 10런)에서 나온 것:

| gs | TPOT(ms) | wasted/total | 뉴런 로드(전송 바이트) | 그룹 이벤트(전송 건수) |
|---:|---:|---:|---:|---:|
| 2 | 178.89 | 34.08% | **35.4M (최소)** | **17.7M (최대)** |
| 16 | 121.65 | 46.79% | **76.4M (최대)** | 4.8M |
| 128 | 117.65 | 50.07% | 58.3M | 0.5M |

gs=2 는 **옮긴 바이트가 가장 적은데 가장 느리다.** 대역폭 가설은 여기서 죽고, 비용은
`이벤트 건수 x 고정비` 쪽에 있다. 그런데 남은 두 후보가 갈리지 않았다.

- (a) 결정 비용 — 매 토큰 도는 DFR EMA -> tau -> `argsort_top_k` -> mask 체인, `n_group` 에 비례
- (b) 전송 비용 — 그룹당 `cudaMemcpyAsync` 한 번, 호출 수는 `1/gs` 에 비례

둘 다 대략 `1/gs` 로 움직여서 **회귀로는 원리적으로 분리되지 않는다**(공선성).
2항 회귀를 돌리면 `TPOT = 112.1ms + 8.0us x 호출수 + 7.24us x n_group` 이 나오지만,
설명변수의 비가 0.63~1.36 사이에서만 흔들려 계수 분할이 식별되지 않는다.

그래서 스텝 시간을 직접 쪼개 재기로 했다.

## 2. 무엇을 재는가

| 버킷 | 내용 | 측정 방식 | 이론적 스케일 |
|---|---|---|---|
| `topk` | `argsort_top_k` ~ mask 생성 | CUDA event | `n_group` 비례, 1024 에서 CUB 폴백 |
| `score` | DFR EMA 갱신 + tau 필터 | CUDA event | **뉴런 수 비례 — gs 와 무관해야 한다** |
| `pcie` | 워커 스레드의 H2D 전송 | 호스트 타이머 | 호출 수 비례 |
| `stall` | 메인이 워커를 실제로 기다린 시간 | 호스트 타이머 | `pcie` 중 노출된 몫 |
| `step_total` | 그래프 1회 = 디코드 토큰 1개 | 호스트 타이머 | 분모 |
| `sparse_launch` | sparse split 발행 비용 | 호스트 타이머 | 대조군 |

`score` 가 평평하고 `topk` 만 커지면 결정 항의 정체가 top-k 로 확정된다.
`score` 까지 커지면 위 모델이 틀린 것이다.

**① compute 는 잔차**로 산출한다: `step_total - stall - topk - score`.

## 3. 아키텍처 상 전제

`SingleThreadExecutor` 를 읽고 확인한 것 — reload 와 sparse 연산은 **겹치지 않고
같은 워커 스레드 큐에서 직렬화된다.**

- `post(kairox_batch_reload, ...)` -> IO 큐에 memcpy 적재
- `make_anchor(...)` -> 앵커를 켜고, 이후 `submit(mul_mat_sparse)` 는 즉시 실행이 아니라
  `anchor->pending` 에 보류
- 워커가 앞의 memcpy 를 다 처리하고 앵커 태스크에 도달해야 pending 이 풀림

겹치는 상대는 메인 그래프 스트림(attention 등)이다. 그래서 호출당 고정비가 그대로
크리티컬 패스에 더해지고, `split_fut.get()` 이 그 노출 지점이 된다.

## 4. 변경 내용

5개 파일, +346 / -9.

### `ggml/include/ggml-kairox.hpp` (+86)

버킷 enum, 원자 누산기, RAII 타이머.

```cpp
const bool k_kairox_profile = get_env_bool("KAIROX_PROFILE", false);

struct kairox_profiler {
    std::atomic<uint64_t> ns[KAIROX_PROF_COUNT];
    std::atomic<uint64_t> n[KAIROX_PROF_COUNT];
    void add(int bucket, uint64_t dt_ns);
};

inline kairox_profiler & kairox_prof() {   // 함수 지역 static = TU 간 단일 인스턴스
    static kairox_profiler prof;
    return prof;
}

struct kairox_prof_scope {                  // 소멸자에서 적립 -> early return 에도 샌다
    int bucket; uint64_t t0;
    ~kairox_prof_scope() { if (k_kairox_profile) kairox_prof().add(bucket, kairox_now_ns() - t0); }
};
```

기본값 false 라 켜지 않으면 baseline 그대로다.

### `src/llama-graph.cpp` (+21)

`build_sparse_ffn_dfr` 의 노드에 이름을 박아 구간을 가른다.
`ffn_dfr_s_*` = 점수 갱신, `ffn_dfr_k_*` = top-k.

```cpp
ggml_format_name(mask,               "ffn_dfr_s_mask-%d", il);
ggml_format_name(dfr_scores,         "ffn_dfr_s_ema-%d", il);
ggml_format_name(threshold_mask,     "ffn_dfr_s_tau-%d", il);
ggml_format_name(filtered_dfr_scores,"ffn_dfr_s_filtered-%d", il);
ggml_format_name(topk_idx,           "ffn_dfr_k_argsort-%d", il);
ggml_format_name(topk_rows,          "ffn_dfr_k_getrows-%d", il);   // 원래는 익명 임시값
...
```

**`cb()` 를 쓰면 안 된다.** `cb()` 는 `cb_func` 가 있을 때 콜백만 부르고 텐서 이름을
설정하지 않는다(`llama-graph.cpp:961`). 처음에 `cb()` 로 붙였다가 `topk` 버킷이 0 으로
나왔다.

`get_rows(group_identity, topk_idx)` 는 원래 익명 임시값이었는데 `topk_rows` 로 꺼내
따로 이름을 줬다. `O(n_cached x n_group)` 라 `argsort` 의 `O(n log n)` 과 스케일이
다르므로, `topk` 가 크게 나오면 이 둘을 갈라 봐야 한다.

### `ggml/src/ggml-cuda/ggml-cuda.cu` (+221)

**(1) CUDA event 링버퍼.** `topk`/`score` 는 메인 스트림 커널이라 호스트 타이머로는
발행 시간만 잡힌다. 노드 앞뒤로 event 를 걸되 **elapsed 를 그 자리에서 읽지 않는다** —
`cudaEventSynchronize` 가 파이프라인을 세워 측정 대상이 오염된다. 2048칸 링에 쌓아두고
한 바퀴 돌아 재사용할 때(이미 끝났을 때) 뒤늦게 걷어낸다.

**(2) 반복 본문 전체를 감싸는 RAII 가드.** 처음엔 `ggml_cuda_compute_forward` 호출만
감쌌는데, 이 루프에는 융합 경로가 여러 개 있고 전부 `continue` 로 빠져나가서 그 노드들을
놓쳤다. 루프 본문 맨 위에 가드를 두어 어떤 경로로 나가든 끝 이벤트가 찍히게 했다.

```cpp
for (int i = 0; i < cgraph->n_nodes; i++) {
    ggml_tensor * node = cgraph->nodes[i];
    kairox_prof_node_guard prof_guard(cuda_ctx->stream(), node);  // 생성자=beg, 소멸자=end
    ...
}
```

**(3) 링버퍼 락.** `KAIROX_PARALLEL` 에서 sparse split 은 워커 스레드가
`ggml_backend_graph_compute_async` 로 실행한다. 즉 이 루프가 메인/워커 두 스레드에서
동시에 돌 수 있어 `next`/`slots` 에 경쟁이 생긴다. `std::mutex` 로 막았다.

**(4) PCIe 계측.** 워커에 `post` 되는 reload 태스크를 타이머 람다로 감쌌다.
`kairox_batch_reload` 가 끝에 `cudaStreamSynchronize` 를 하므로 실제 H2D 시간이 잡힌다.
gather 경로도 같은 자로 재도록 감쌌다.

### `ggml/src/ggml-backend.cpp` (+24)

`split_fut.get()` 3곳을 `kairox_prof_fut_get()` 으로 교체 = `stall`.
`ggml_backend_sched_compute_splits` 전체를 `kairox_prof_scope` 로 감쌈 = `step_total`.

### `src/llama-kairox.cpp` (+3)

`kairox_cache_manager` 소멸자에서 `kairox_profile_dump()` 호출. activation 덤프와 같은
지점이라 **정상 종료해야 파일이 남는다**(Ctrl+C 로 끊으면 안 남는다).

## 5. 사용법

```sh
KAIROX_PROFILE=1 KAIROX_PROFILE_PATH=prof.csv <기존 실행 명령 그대로>
```

- `KAIROX_PROFILE_DEBUG=1` — CUDA 디스패치를 지나가는 `ffn_*` 노드 이름을 400개까지
  출력한다. 버킷이 0 으로 남을 때 이름 문제인지 백엔드 배정 문제인지 가리는 용도.

출력 (`bucket,calls,total_ms,per_step_ms,per_call_us`).

## 6. 검증 결과

3070 / vb=6 / gs=16 / 8토큰:

```
bucket,calls,total_ms,per_step_ms,per_call_us
topk,512,3.696,0.461980,7.218
score,768,7.404,0.925552,9.641
pcie,6438,594.788,74.348438,92.387
stall,512,110.151,13.768934,215.140
step_total,8,1171.228,146.403503,146403.503
sparse_launch,512,5.267,0.658330,10.286
```

- 6개 버킷 모두 값이 잡힌다. `step_total` 146ms/step 과 `eval time` 134ms/token 이 정합.
- 계측 오버헤드는 노이즈에 묻힌다 (같은 조건 `PROFILE=0` 1200ms/tok vs `PROFILE=1`
  1063ms/tok — 둘 다 호스트 부하가 심할 때 잰 값이라 차이가 역전되어 나왔다).
- **첫 인상**: gs=16 에서 `topk`+`score` 는 1.4ms/step 으로 146ms 중 1% 수준이고,
  워커의 `pcie` 74ms 중 실제 노출은 `stall` 13.8ms 다. 다만 8토큰짜리 웜업 전이 구간이라
  결론으로 쓰면 안 된다.

## 7. 한계 / 남은 일

1. **융합된 노드는 자기 반복을 갖지 않는다.** 노드 i 가 i+1, i+2 를 흡수하면 그 둘은
   루프 반복 자체가 없어 가드도 없다. 그 시간은 융합한 노드의 버킷에 들어가거나, 융합한
   노드가 미분류면 누락된다. `topk` 가 레이어당 7개 중 2개, `score` 가 4개 중 3개만
   잡히는 이유가 이것으로 보인다 — **절대값을 그대로 믿지 말고 gs 축 상대 비교로 쓸 것.**
2. `step_total` 에 prefill 호출도 섞인다. bench 10런이면 prefill 10회 vs decode 5110회라
   무시할 수준이지만 단발 실행에서는 감안해야 한다.
3. `topk`+`score`+`pcie` 의 합은 `step_total` 을 넘을 수 있다(메인 스트림과 워커가 병렬).
   `stall` 이 그중 노출된 몫이라는 해석이 전제다.
4. 분류기에 `ffn_load_group`/`ffn_evict_group` 매칭이 남아 있는데, 개명 이후 죽은 코드다.
5. 아직 **gs 축 스윕을 돌리지 않았다.** 다음 단계는 gs 2~128 x `KAIROX_GATHER` on/off 로
   `score` 가 평평한지, `topk` 가 1024 경계에서 뛰는지, `pcie` 가 호출 수를 따라가는지를
   보는 것이다.

## 8. 전체 diff

```diff
diff --git a/ggml/include/ggml-kairox.hpp b/ggml/include/ggml-kairox.hpp
index 483fc7ef6..8070693f6 100644
--- a/ggml/include/ggml-kairox.hpp
+++ b/ggml/include/ggml-kairox.hpp
@@ -3,6 +3,8 @@
 #include "ggml.h"
 
 #include <algorithm>
+#include <atomic>
+#include <chrono>
 #include <cstdint>
 #include <cstdlib>
 #include <cstring>
@@ -78,6 +80,90 @@ const int  k_kairox_reload_window      = std::max(1, get_env_int("KAIROX_RELOAD_
 // 매우 느리다. 검증 전용.
 const bool k_kairox_gather_verify      = get_env_bool("KAIROX_GATHER_VERIFY", false);
 
+// --- 스텝 시간 분해 계측 (KAIROX_PROFILE) -----------------------------------
+// group_size 스윕에서 TPOT 가 1/gs 에 비례해 나빠지는 것은 확인됐지만, 전송 호출 수와
+// n_group 이 둘 다 1/gs 로 움직여서 회귀로는 어느 항이 지배적인지 분리되지 않는다.
+// 그래서 디코드 스텝 시간을 네 갈래로 직접 잰다.
+//
+//   ① compute : sparse FFN 커널 등 실제 연산            (잔차로 산출)
+//   ② topk    : argsort_top_k ~ mask 생성 (n_group 에 비례, 1024 에서 CUB 폴백)
+//   ③ score   : DFR EMA 갱신 + tau 필터    (뉴런 수에 비례하므로 gs 와 무관해야 한다)
+//   ④ pcie    : 워커 스레드의 H2D 전송     (호출 수에 비례)
+//   ⑤ stall   : 메인 스레드가 워커를 실제로 기다린 시간
+//
+// ②③ 는 메인 CUDA 스트림의 커널이라 호스트 타이머로 잴 수 없어 CUDA event 로 재고,
+// ④⑤ 는 호스트 측 대기라 steady_clock 으로 잰다. 메인 스트림과 워커가 병렬로 돌기
+// 때문에 ②+③+④ 의 합은 스텝 시간을 넘을 수 있다 — ⑤ 가 그중 실제로 노출된 몫이다.
+const bool k_kairox_profile = get_env_bool("KAIROX_PROFILE", false);
+
+enum kairox_prof_bucket {
+    KAIROX_PROF_TOPK = 0,
+    KAIROX_PROF_SCORE,
+    KAIROX_PROF_PCIE,
+    KAIROX_PROF_STALL,
+    KAIROX_PROF_STEP,        // 그래프 1회(= 디코드 토큰 1개) 벽시계 시간
+    KAIROX_PROF_SPARSE_LAUNCH,  // 워커에서 sparse split 을 발행하는 데 든 시간
+    KAIROX_PROF_COUNT
+};
+
+inline const char * kairox_prof_name(int b) {
+    switch (b) {
+        case KAIROX_PROF_TOPK:          return "topk";
+        case KAIROX_PROF_SCORE:         return "score";
+        case KAIROX_PROF_PCIE:          return "pcie";
+        case KAIROX_PROF_STALL:         return "stall";
+        case KAIROX_PROF_STEP:          return "step_total";
+        case KAIROX_PROF_SPARSE_LAUNCH: return "sparse_launch";
+        default:                        return "unknown";
+    }
+}
+
+// 여러 번역 단위에서 같은 인스턴스를 봐야 하므로 함수 지역 static 으로 둔다.
+struct kairox_profiler {
+    std::atomic<uint64_t> ns[KAIROX_PROF_COUNT];
+    std::atomic<uint64_t> n[KAIROX_PROF_COUNT];
+
+    kairox_profiler() {
+        for (int i = 0; i < KAIROX_PROF_COUNT; ++i) {
+            ns[i].store(0);
+            n[i].store(0);
+        }
+    }
+
+    void add(int bucket, uint64_t dt_ns) {
+        ns[bucket].fetch_add(dt_ns, std::memory_order_relaxed);
+        n[bucket].fetch_add(1, std::memory_order_relaxed);
+    }
+};
+
+inline kairox_profiler & kairox_prof() {
+    static kairox_profiler prof;
+    return prof;
+}
+
+inline uint64_t kairox_now_ns() {
+    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
+               std::chrono::steady_clock::now().time_since_epoch())
+        .count();
+}
+
+// 구간 하나를 재서 버킷에 적립하는 헬퍼. 소멸자에서 적립하므로 early return 이 있어도 샌다.
+struct kairox_prof_scope {
+    int      bucket;
+    uint64_t t0;
+
+    explicit kairox_prof_scope(int bucket) : bucket(bucket), t0(k_kairox_profile ? kairox_now_ns() : 0) {}
+
+    ~kairox_prof_scope() {
+        if (k_kairox_profile) {
+            kairox_prof().add(bucket, kairox_now_ns() - t0);
+        }
+    }
+};
+
+// 계측 결과를 CSV 로 남긴다. KAIROX_DUMP_ACTIVATION 과 같이 종료 시점에 한 번 부른다.
+void kairox_profile_dump();
+
 /**
  * 캐시 관리 정책을 실제로 수행하는 구조체
  * 각 레이어마다 하나씩 존재하며 , 레이어의 FFN 가중치를 메모리로 로드하고
diff --git a/ggml/src/ggml-backend.cpp b/ggml/src/ggml-backend.cpp
index df95c0386..19d86404f 100644
--- a/ggml/src/ggml-backend.cpp
+++ b/ggml/src/ggml-backend.cpp
@@ -1557,10 +1557,26 @@ void kairox_register_dependency(ggml_backend_sched_t        sched,
     kairox_append_event(sched, dst, dst_state, event);
 }
 
+// KAIROX_PROFILE: future 를 기다리는 이 지점이 메인 스레드가 워커(전송 + sparse 연산)를
+// 실제로 붙잡혀 있는 곳이다. 여기서 걸린 시간이 곧 TPOT 에 노출된 몫이다.
+static enum ggml_status kairox_prof_fut_get(std::future<enum ggml_status> & fut) {
+    if (!k_kairox_profile) {
+        return fut.get();
+    }
+
+    const uint64_t   t0 = kairox_now_ns();
+    enum ggml_status ec = fut.get();
+    kairox_prof().add(KAIROX_PROF_STALL, kairox_now_ns() - t0);
+    return ec;
+}
+
 static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
     GGML_ASSERT(sched);
     struct ggml_backend_sched_split * splits = sched->splits;
 
+    // 그래프 1회 = 디코드 토큰 1개. 다른 구간들의 분모가 된다.
+    kairox_prof_scope prof_step(KAIROX_PROF_STEP);
+
     ggml_tensor * prev_ids_tensor = nullptr;
     std::vector<int32_t> ids;
     std::vector<ggml_bitset_t> used_ids;
@@ -1574,7 +1590,7 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
 
         if (k_enable_kairox_parallel && split_fut.valid() &&
             ggml_backend_dev_type(split_backend->device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
-            enum ggml_status async_ec = split_fut.get();
+            enum ggml_status async_ec = kairox_prof_fut_get(split_fut);
             if (async_ec != GGML_STATUS_SUCCESS) {
                 return async_ec;
             }
@@ -1589,7 +1605,7 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
                     for (int k = 0; k < kairox_extra->event_count; ++k) {
                         if (kairox_extra->states[k] == KAIROX_EVENT_SYNCHRONIZE) {
                             if (!waited_for_async_ec && split_fut.valid() && async_split_fut_flag == KAIROX_SPLIT_AXPY_SPARSE) {
-                                enum ggml_status async_ec = split_fut.get();
+                                enum ggml_status async_ec = kairox_prof_fut_get(split_fut);
                                 if (async_ec != GGML_STATUS_SUCCESS) {
                                     return async_ec;
                                 }
@@ -1737,10 +1753,12 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
         if (!sched->callback_eval) {
             auto * kairox_extra = (kairox_tensor_extra *) split->graph.nodes[0]->extra;
             if (k_enable_kairox_parallel && kairox_extra && kairox_extra->split_flag == KAIROX_SPLIT_MUL_MAT_SPARSE) {
+                kairox_prof_scope prof_launch(KAIROX_PROF_SPARSE_LAUNCH);
                 split_fut = sched->kairox_executor->submit(SingleThreadExecutor::KairoxWaitType::KAIROX_WAIT_MUL_MAT_SPARSE,
                                                          ggml_backend_graph_compute_async, split_backend, &split->graph);
                 async_split_fut_flag = KAIROX_SPLIT_MUL_MAT_SPARSE;
             } else if (k_enable_kairox_parallel && kairox_extra && kairox_extra->split_flag == KAIROX_SPLIT_AXPY_SPARSE) {
+                kairox_prof_scope prof_launch(KAIROX_PROF_SPARSE_LAUNCH);
                 split_fut = sched->kairox_executor->submit(SingleThreadExecutor::KairoxWaitType::KAIROX_WAIT_AXPY_SPARSE,
                                                          ggml_backend_graph_compute_async, split_backend, &split->graph);
                 async_split_fut_flag = KAIROX_SPLIT_AXPY_SPARSE;
@@ -1793,7 +1811,7 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
     }
 
     if (k_enable_kairox_parallel && split_fut.valid()) {
-        enum ggml_status async_ec = split_fut.get();
+        enum ggml_status async_ec = kairox_prof_fut_get(split_fut);
         if (async_ec != GGML_STATUS_SUCCESS) {
             return async_ec;
         }
diff --git a/ggml/src/ggml-cuda/ggml-cuda.cu b/ggml/src/ggml-cuda/ggml-cuda.cu
index d97e055c5..d0dec8f0a 100644
--- a/ggml/src/ggml-cuda/ggml-cuda.cu
+++ b/ggml/src/ggml-cuda/ggml-cuda.cu
@@ -2649,6 +2649,195 @@ static void ggml_cuda_reload_plan(ggml_backend_cuda_context & ctx, ggml_tensor *
     CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
 }
 
+// --- KAIROX_PROFILE: 메인 스트림 커널 구간 계측 -------------------------------
+//
+// topk/score 구간은 메인 CUDA 스트림의 커널이라 호스트 타이머로는 발행 시간만 잡힌다.
+// 노드 앞뒤로 CUDA event 를 걸어 실제 GPU 시간을 잰다.
+//
+// elapsed 를 그 자리에서 읽으면 cudaEventSynchronize 가 파이프라인을 세워 측정 대상이
+// 오염되므로, 이벤트 쌍을 링버퍼에 쌓아두고 한 바퀴 돌아 재사용할 때(=이미 끝났을 때)
+// 뒤늦게 읽어서 적립한다.
+struct kairox_prof_gpu {
+    struct slot {
+        cudaEvent_t beg    = nullptr;
+        cudaEvent_t end    = nullptr;
+        int         bucket = -1;   // -1 이면 비어 있음
+    };
+
+    static constexpr size_t n_slots = 2048;
+
+    std::vector<slot> slots;
+    size_t            next = 0;
+    // KAIROX_PARALLEL 에서 sparse split 은 워커 스레드가 ggml_backend_graph_compute_async 로
+    // 실행한다. 즉 이 링은 메인/워커 두 스레드에서 동시에 불릴 수 있어 락이 필요하다.
+    std::mutex mtx;
+
+    kairox_prof_gpu() : slots(n_slots) {}
+
+    ~kairox_prof_gpu() { drain(); }
+
+    // 사용할 슬롯을 확보한다. 이미 쓰던 슬롯이면 결과를 먼저 걷어낸다.
+    slot * acquire(int bucket) {
+        std::lock_guard<std::mutex> lock(mtx);
+
+        slot & sl = slots[next];
+        next      = (next + 1) % n_slots;
+
+        harvest_locked(sl);
+
+        if (!sl.beg || !sl.end) {
+            // 한쪽만 만들어진 상태로 남지 않게 둘 다 확보되었을 때만 슬롯을 내준다.
+            if (!sl.beg && cudaEventCreate(&sl.beg) != cudaSuccess) {
+                return nullptr;
+            }
+            if (!sl.end && cudaEventCreate(&sl.end) != cudaSuccess) {
+                return nullptr;
+            }
+        }
+
+        sl.bucket = bucket;
+        return &sl;
+    }
+
+    // 끝난 구간의 시간을 버킷에 적립한다. 호출자가 mtx 를 들고 있어야 한다.
+    void harvest_locked(slot & sl) {
+        if (sl.bucket < 0) {
+            return;
+        }
+
+        // 링을 한 바퀴 돈 뒤라 대개 이미 끝나 있다. 아직이면 여기서 기다린다.
+        if (cudaEventSynchronize(sl.end) == cudaSuccess) {
+            float ms = 0.0f;
+            if (cudaEventElapsedTime(&ms, sl.beg, sl.end) == cudaSuccess) {
+                kairox_prof().add(sl.bucket, (uint64_t) (ms * 1e6f));
+            }
+        }
+
+        sl.bucket = -1;
+    }
+
+    void drain() {
+        std::lock_guard<std::mutex> lock(mtx);
+
+        for (auto & sl : slots) {
+            harvest_locked(sl);
+            if (sl.beg) {
+                cudaEventDestroy(sl.beg);
+                sl.beg = nullptr;
+            }
+            if (sl.end) {
+                cudaEventDestroy(sl.end);
+                sl.end = nullptr;
+            }
+        }
+    }
+};
+
+static kairox_prof_gpu & kairox_prof_gpu_ring() {
+    static kairox_prof_gpu ring;
+    return ring;
+}
+
+// 노드 이름으로 어느 버킷인지 정한다. cb() 가 "<name>-<il>" 로 붙이므로 접두사로 본다.
+// 이름이 없거나 관심 밖이면 -1.
+static int kairox_prof_bucket_of(const ggml_tensor * node) {
+    if (!node || node->name[0] == '\0') {
+        return -1;
+    }
+    if (strncmp(node->name, "ffn_dfr_k_", 10) == 0 ||
+        strncmp(node->name, "ffn_load_group", 14) == 0 ||
+        strncmp(node->name, "ffn_evict_group", 15) == 0) {
+        return KAIROX_PROF_TOPK;
+    }
+    if (strncmp(node->name, "ffn_dfr_s_", 10) == 0) {
+        return KAIROX_PROF_SCORE;
+    }
+    return -1;
+}
+
+// 진단용: 어떤 노드가 CUDA 디스패치를 지나가는지 한 번만 훑어본다.
+// 이름이 안 붙거나 다른 백엔드로 배정되면 버킷이 0 으로 남으므로 그 원인을 찾는 데 쓴다.
+static void kairox_prof_debug_node(const ggml_tensor * node) {
+    static const bool enabled = get_env_bool("KAIROX_PROFILE_DEBUG", false);
+    static int        left    = 400;
+
+    if (!enabled || left <= 0) {
+        return;
+    }
+    if (strncmp(node->name, "ffn_", 4) == 0) {
+        GGML_LOG_INFO("kairox_prof_debug: node='%s' op=%s\n", node->name, ggml_op_name(node->op));
+        --left;
+    }
+}
+
+// 노드 하나를 감싸는 RAII 가드. 생성자에서 시작 이벤트를, 소멸자에서 끝 이벤트를 찍는다.
+// 관심 밖 노드(분류기가 -1)면 아무것도 하지 않는다.
+struct kairox_prof_node_guard {
+    kairox_prof_gpu::slot * slot   = nullptr;
+    cudaStream_t            stream = nullptr;
+
+    kairox_prof_node_guard(cudaStream_t stream, const ggml_tensor * node) : stream(stream) {
+        if (!k_kairox_profile) {
+            return;
+        }
+
+        kairox_prof_debug_node(node);
+
+        const int bucket = kairox_prof_bucket_of(node);
+        if (bucket < 0) {
+            return;
+        }
+
+        slot = kairox_prof_gpu_ring().acquire(bucket);
+        if (slot) {
+            CUDA_CHECK(cudaEventRecord(slot->beg, stream));
+        }
+    }
+
+    ~kairox_prof_node_guard() {
+        if (slot) {
+            CUDA_CHECK(cudaEventRecord(slot->end, stream));
+        }
+    }
+
+    kairox_prof_node_guard(const kairox_prof_node_guard &)             = delete;
+    kairox_prof_node_guard & operator=(const kairox_prof_node_guard &) = delete;
+};
+
+void kairox_profile_dump() {
+    if (!k_kairox_profile) {
+        return;
+    }
+
+    kairox_prof_gpu_ring().drain();
+
+    const char * path_env = getenv("KAIROX_PROFILE_PATH");
+    const std::string path = path_env ? path_env : "kairox_profile.csv";
+
+    FILE * f = fopen(path.c_str(), "w");
+    if (!f) {
+        GGML_LOG_WARN("%s: '%s' 를 열지 못했다\n", __func__, path.c_str());
+        return;
+    }
+
+    auto & prof = kairox_prof();
+
+    const uint64_t steps = prof.n[KAIROX_PROF_STEP].load();
+
+    fprintf(f, "bucket,calls,total_ms,per_step_ms,per_call_us\n");
+    for (int i = 0; i < KAIROX_PROF_COUNT; ++i) {
+        const uint64_t ns    = prof.ns[i].load();
+        const uint64_t calls = prof.n[i].load();
+
+        fprintf(f, "%s,%llu,%.3f,%.6f,%.3f\n", kairox_prof_name(i), (unsigned long long) calls, ns / 1e6,
+                steps ? ns / 1e6 / steps : 0.0, calls ? ns / 1e3 / calls : 0.0);
+    }
+    fclose(f);
+
+    GGML_LOG_INFO("%s: wrote step profile to '%s' (steps=%llu)\n", __func__, path.c_str(),
+                  (unsigned long long) steps);
+}
+
 static void kairox_batch_reload(char *        weight_base,
                                     char *        cache_base,
                                     size_t        nbytes,
@@ -2703,16 +2892,34 @@ static void ggml_cuda_reload_exec(ggml_backend_cuda_context & ctx, ggml_tensor *
 
     auto * kairox_extra    = (kairox_tensor_extra *) dst->extra;
     auto * kairox_executor = (SingleThreadExecutor *) kairox_extra->kairox_executor;
+    // 전송 태스크는 워커 스레드에서 돌고 끝에 cudaStreamSynchronize 를 하므로,
+    // 호스트 타이머로 감싸면 실제 H2D 시간이 잡힌다. post() 는 함수+인자를 묶어 큐에
+    // 넣으므로, 미리 묶어둔 것을 타이머 람다로 한 번 더 감싸 넘긴다.
+    const auto post_timed = [&](auto && bound) {
+        if (k_kairox_profile) {
+            kairox_executor->post([bound = std::move(bound)]() mutable {
+                const uint64_t t0 = kairox_now_ns();
+                bound();
+                kairox_prof().add(KAIROX_PROF_PCIE, kairox_now_ns() - t0);
+            });
+        } else {
+            kairox_executor->post(std::move(bound));
+        }
+    };
+
     if (k_kairox_gather) {
         // 결정 단위는 그대로 두고 전송만 묶는다. 호출 횟수가 reload_count 개에서 3개로 줄어든다.
-        kairox_executor->post(kairox_gather_reload, weight_base, cache_base, group_nbytes, cudaStreamPerThread,
-                              (const reload_pair *) kairox_lc->reload_plan.data(), kairox_lc->reload_count);
+        post_timed(SingleThreadExecutor::make_bound(kairox_gather_reload, weight_base, cache_base, group_nbytes,
+                                                    cudaStreamPerThread,
+                                                    (const reload_pair *) kairox_lc->reload_plan.data(),
+                                                    kairox_lc->reload_count));
     } else {
         for (size_t window_offset = 0; window_offset < kairox_lc->reload_count;) {
             size_t window_size = MIN(kairox_lc->reload_window_size, kairox_lc->reload_count - window_offset);
 
-            kairox_executor->post(kairox_batch_reload, weight_base, cache_base, group_nbytes,
-                                cudaStreamPerThread, window_offset, window_size, kairox_lc->reload_plan.data());
+            post_timed(SingleThreadExecutor::make_bound(kairox_batch_reload, weight_base, cache_base, group_nbytes,
+                                                        cudaStreamPerThread, window_offset, window_size,
+                                                        kairox_lc->reload_plan.data()));
 
             window_offset += window_size;
         }
@@ -4119,6 +4326,12 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
 
             for (int i = 0; i < cgraph->n_nodes; i++) {
                 ggml_tensor * node = cgraph->nodes[i];
+
+                // KAIROX_PROFILE: 이 루프에는 융합 경로가 여러 개 있고 전부 continue 로
+                // 빠져나가므로, 디스패치 호출만 감싸면 그 노드들을 놓친다. 반복 본문 전체를
+                // RAII 로 감싸 어떤 경로로 나가든 끝 이벤트가 기록되게 한다.
+                kairox_prof_node_guard prof_guard(cuda_ctx->stream(), node);
+
                 if (is_concurrent_event_active) {
                     GGML_ASSERT(concurrent_event);
 
diff --git a/src/llama-graph.cpp b/src/llama-graph.cpp
index ff9ad78ab..152d584ec 100644
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -1298,9 +1298,12 @@ void llm_graph_context::build_sparse_ffn_dfr(kairox_layer_cache * lc,
                                              float                    threshold,
                                              int                      il) const {
     ggml_tensor * mask = ggml_shifted_step(ctx0, lc->sparse_idx, -threshold, false);
+    // KAIROX_PROFILE 이 노드 이름 접두사로 구간을 가른다 (ffn_dfr_s_ = 점수 갱신, ffn_dfr_k_ = top-k).
+    ggml_format_name(mask, "ffn_dfr_s_mask-%d", il);
     // sparse_idx는 predictor의 sigmoid 출력. 0.5보다 크면 1, 아니면 0으로 mask를 만든다.
     if (lc->sparse_idx->ne[1] > 1) {
         mask = ggml_sum_cols(ctx0, mask);
+        ggml_format_name(mask, "ffn_dfr_s_masksum-%d", il);
     }
 
     /**
@@ -1314,6 +1317,7 @@ void llm_graph_context::build_sparse_ffn_dfr(kairox_layer_cache * lc,
         ctx0, ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, mask, lc->cache_shape.group_size, lc->cache_shape.n_groups)));
     ggml_tensor * dfr_scores = ggml_scale_add(ctx0, lc->dfr_scores, deltas, (float *) lc->dfr_ema_coeffs->data,
                                               (float) lc->sparse_idx->ne[1] * lc->cache_shape.group_size, true);
+    ggml_format_name(dfr_scores, "ffn_dfr_s_ema-%d", il);
     /**
     * 논문 알고리즘에는 One-Hit Wonder 현상을 막기 위해서 tau 값을 주고있다.
     * One-Hit Wonder 현상은 어떤 뉴런이 한 번만 활성화되었는데, EMA 값은 높게 나와서 계속 top-k에 포함되는 현상이다.
@@ -1327,26 +1331,39 @@ void llm_graph_context::build_sparse_ffn_dfr(kairox_layer_cache * lc,
     const float tau = (1.0f - *(float *) lc->dfr_ema_coeffs->data) + 1e-6f;
     // mask 값이 0이면 topk에 포함되지 않도록
     ggml_tensor * threshold_mask = ggml_shifted_step(ctx0, dfr_scores, -tau, false);
+    ggml_format_name(threshold_mask, "ffn_dfr_s_tau-%d", il);
     ggml_tensor * filtered_dfr_scores = ggml_mul(ctx0, dfr_scores, threshold_mask);
+    ggml_format_name(filtered_dfr_scores, "ffn_dfr_s_filtered-%d", il);
 
     // 계산된 DFR 점수에 따라 top-k 그룹을 선택. argsort() API 사용 !
     // 이 때 k 값은 VRAM capacity이다, 즉 존재하는 그룹들 중 DFR Score에 따라 top k 그룹들이 VRAM으로 load 되는 것
     ggml_tensor * topk_idx   = ggml_argsort_top_k(ctx0, filtered_dfr_scores, lc->cache_shape.n_cached_groups);
+    ggml_format_name(topk_idx, "ffn_dfr_k_argsort-%d", il);
     // top-k 그룹에 대한 마스크 생성
-    ggml_tensor * topk_mask  = ggml_sum_cols(ctx0, ggml_get_rows(ctx0, kairox_cm->group_identity, topk_idx));
+    // get_rows 는 n_cached_groups x n_group 을 훑으므로 argsort(n log n) 와 스케일이 다르다.
+    // 둘 다 ffn_dfr_k_ 로 묶여 있으니, topk 가 크게 나오면 이 둘을 따로 갈라 봐야 한다.
+    ggml_tensor * topk_rows  = ggml_get_rows(ctx0, kairox_cm->group_identity, topk_idx);
+    ggml_format_name(topk_rows, "ffn_dfr_k_getrows-%d", il);
+    ggml_tensor * topk_mask  = ggml_sum_cols(ctx0, topk_rows);
+    ggml_format_name(topk_mask, "ffn_dfr_k_mask-%d", il);
 
     ggml_tensor * diff_mask  = ggml_xor(ctx0, lc->group_mask, topk_mask);
+    ggml_format_name(diff_mask, "ffn_dfr_k_diff-%d", il);
 
     // load group tensor 완성
     load_group = ggml_and(ctx0, topk_mask, diff_mask);
     cb(load_group, "ffn_load_group", il);
+    ggml_format_name(load_group, "ffn_dfr_k_load-%d", il);
     ggml_build_forward_expand(gf, load_group);
 
     // evict group tensor 완성
     evict_group = ggml_and(ctx0, lc->group_mask, diff_mask);
     cb(evict_group, "ffn_evict_group", il);
+    ggml_format_name(evict_group, "ffn_dfr_k_evict-%d", il);
     ggml_build_forward_expand(gf, evict_group);
-    ggml_build_forward_expand(gf, ggml_cpy(ctx0, topk_mask, lc->group_mask));
+    ggml_tensor * mask_cpy = ggml_cpy(ctx0, topk_mask, lc->group_mask);
+    ggml_format_name(mask_cpy, "ffn_dfr_k_cpy-%d", il);
+    ggml_build_forward_expand(gf, mask_cpy);
 }
 
 ggml_tensor * llm_graph_context::build_sparse_ffn_hidden(ggml_tensor *& cur_up,
diff --git a/src/llama-kairox.cpp b/src/llama-kairox.cpp
index 47958c27b..675ec3b4a 100644
--- a/src/llama-kairox.cpp
+++ b/src/llama-kairox.cpp
@@ -496,6 +496,9 @@ static void kairox_dump_activation_csv(const std::vector<kairox_layer_cache *>
 
 
 kairox_cache_manager::~kairox_cache_manager() {
+    // 스텝 시간 분해 결과(KAIROX_PROFILE)도 같은 시점에 떨군다.
+    kairox_profile_dump();
+
     if (k_kairox_dump_activation) {
         kairox_dump_activation_csv(layer_caches);
     }
```
