# Adaptive Neuron Balancer (ANB) 구현

논문 *"Kairox: Adaptive GPU-CPU Hybrid LLM Inference via Online Neuron Balancing"* (OSDI '26)
Algorithm 1 중 **Phase 1 (Adaptive Balancing-Intensity Control)** 을 구현한 내용이다.

---

## 1. 작업 전 상태 — 논문 Algorithm 1 대비 구현/미구현

### Phase 2 (Momentum Update & Group Filtering) — 이미 구현돼 있었음

| Alg.1 | 논문 | 코드 |
|---|---|---|
| 7 | `A_t^(g) <- ActivationGrouping(...)` | `llama-graph.cpp:1300-1314` `shifted_step(sparse_idx, -0.5)` -> `reshape` -> `sum_rows` -> `transpose` |
| 11 | `S <- lambda*S + (1-lambda)*A` | `llama-graph.cpp:1315` `ggml_scale_add(dfr_scores, deltas, dfr_ema_coeffs, ...)` |
| 8 | `tau_load <- (1-lambda) + eps` | `llama-graph.cpp:1327` (커밋 `d5a6d17`) |
| 12-15 | one-hit wonder 필터 -> `C_cand` | `llama-graph.cpp:1329-1330` `shifted_step` + `mul` |
| 16 | `TopK(C_cand, K)` | `llama-graph.cpp:1334` `argsort_top_k` |
| 17-18 | `G_load`, `G_evict` | `llama-graph.cpp:1336-1348` XOR + AND |
| 19 | `IssueAsyncIO` | `llama-kairox.cpp:44-118`, `ggml-cuda.cu:2705-2719` |

### Phase 1 (Adaptive Balancing-Intensity Control) — 전부 미구현이었음

| Alg.1 | 논문 | 작업 전 코드 |
|---|---|---|
| 1 | `Feedback <- GetSystemBottleneck()` | 병목 판정 코드 없음 (`IO_BOUND`/`CPU_BOUND` grep 0건) |
| 3 | `lambda <- min(lambda*(1+alpha), lambda_max)` | 없음 |
| 5 | `lambda <- max(lambda*(1-alpha), lambda_min)` | 없음 |
| - | `lambda_min`, `lambda_max` | 상수/환경변수 자체가 없음 |
| - | alpha (조정률) | `KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05` 는 있었으나 **lambda 가 아닌 다른 변수에 적용** |

대신 논문에 없는 대체 구현이 들어가 있었다 (`ggml-kairox.hpp:222-231`):

```cpp
// For simplicity, decrease the maximum load directly when reloading.
int nxt = cur * (1.0f + (to_move.empty() ? +adapt_rate : -adapt_rate));
dfr_clamp_k->store(clamp(nxt, 1, n_cached_groups));
```

lambda 대신 **스왑 예산 `dfr_clamp_k`** (스텝당 교체 가능한 그룹 수 상한, `llama-kairox.cpp:70` 에서 사용)를
+-5% 씩 조절한다. 피드백 신호(`anchor->pending` 이 비었는지)는 이미 있었지만 **lambda 가 아닌 K 에 연결**돼 있었다.

---

## 2. 변경 파일

```
 README.md                       |  51 +++++++++++++++
 ggml/include/ggml-kairox.hpp    | 123 ++++++++++++++++++++++++++++++----
 ggml/include/ggml.h             |   7 +++
 ggml/src/ggml-cuda/ggml-cuda.cu |   6 +-
 ggml/src/ggml-cuda/unary.cu     |   8 +++
 ggml/src/ggml.c                 |  21 ++++++
 src/llama-graph.cpp             |  10 ++-
 src/llama-kairox.cpp            |  41 ++++++++++++
 test_kairox.sh                  |  33 ++++++++-
 9 files changed, 282 insertions(+), 18 deletions(-)
```

---

## 3. 설계 판단 3가지

### 3-1. `GetSystemBottleneck()` — 기존 anchor 신호를 재사용

논문은 "monitors pipeline stalls" 라고만 쓰고 구체적인 방법을 주지 않는다.
`SingleThreadExecutor` 에 이미 그 정보가 있다.

- `post()` 로 들어간 reload memcpy 들은 `io_tasks_` 에 쌓이고, anchor 마커도 그 뒤에 들어간다.
- 워커 루프는 `tasks_`(compute)를 `io_tasks_`(I/O)보다 **먼저** 처리한다.
- `submit()` 된 sparse compute split 은 anchor 가 `active` 인 동안 `anchor->pending` 에 들어간다.

따라서 anchor 가 풀리는 시점에 `pending` 이 비어있지 않다는 것은
**"reload I/O 가 아직 드레인되는 중에 sparse compute split 이 도착했다"** = 계산이 전송을 기다렸다 = `IO_BOUND`.
비어있으면 전송이 먼저 끝났다 = `CPU_BOUND`.

레이어마다 FFN_DOWN anchor 에서 한 번씩 판정하므로, 논문의 "per decoding step, per layer" 와 맞는다.

### 3-2. tau 를 그래프에 상수로 구우면 안 된다

`tau_load = (1 - lambda) + eps` 인데 ANB 가 켜지면 lambda 가 스텝마다 바뀐다.
그런데 llama.cpp 는 디코드 중 그래프를 재사용한다 (512 토큰 실행에서 `graphs reused = 269` 실측).
기존 코드는 빌드 시점 값을 `ggml_shifted_step` 의 `op_params` 에 복사하므로 **첫 스텝 값에 고정**된다.

TAM 업데이트(`ggml_scale_add`)는 이미 op_params 에 호스트 포인터를 저장해 실행 시점에 역참조하는 방식이라
같은 패턴을 shifted_step 에도 추가했다 (`ggml_shifted_step_dyn`).

### 3-3. lambda = 0 예외 (neuralink 프로파일)

`neuralink` 백엔드는 `lambda = 0.00` 이다. 이때 `S = 0*S + 1*A = A`, 그리고 A 는
`deltas / (n_tokens * group_size)` 이므로 `A <= 1`.
반면 `tau = (1 - 0) + eps = 1 + eps` 라서 **어떤 그룹도 필터를 통과하지 못한다** -> 캐시가 정적으로 굳는다.

관성이 없으면 매 스텝이 독립적인 결정이라 one-hit wonder 라는 개념 자체가 성립하지 않으므로,
`lambda <= 0` 이면 필터를 끈다. (아래 6장에서 실측으로 확인)

---

## 4. 파일별 변경 내용

### 4-1. `ggml/include/ggml-kairox.hpp` — ANB 본체

**(a) 설정과 tau 유도식 (Algorithm 1 line 8)**

```cpp
const bool  k_enable_kairox_anb = get_env_bool("KAIROX_ANB", true);
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
```

**(b) `kairox_layer_cache` 필드 추가**

```cpp
ggml_tensor * dfr_ema_coeffs = nullptr; // {lambda, 1 - lambda, normalizer}. ANB 가 앞의 두 값을 매 스텝 갱신한다

/**
 * -tau_load. ggml_shifted_step_dyn 이 실행 시점에 이 주소를 읽는다.
 * 커널이 (x + threshold) > 0 을 계산하므로 부호를 뒤집어 저장한다 -- 즉 score > tau_load.
 * lambda 가 ANB 로 바뀔 때마다 같이 갱신되며, 그래프가 재사용돼도 최신 값이 반영된다.
 */
float dfr_neg_tau = 0.0f;

// ANB 궤적(KAIROX_ANB_TRACE)
std::vector<float>   dbg_anb_lambda;
std::vector<uint8_t> dbg_anb_io_bound;
std::vector<int32_t> dbg_anb_reloads;
```

**(c) Phase 1 본체 — Algorithm 1 line 1-6**

```cpp
inline void kairox_anb_feedback(kairox_layer_cache * lc, bool io_bound) {
    auto * coeffs = (float *) lc->dfr_ema_coeffs->data;  // {lambda, 1 - lambda, normalizer}

    const float alpha  = k_kairox_dfr_lambda_adapt_rate;
    const float lambda = io_bound ? std::min(coeffs[0] * (1.0f + alpha), k_kairox_lambda_max) :
                                    std::max(coeffs[0] * (1.0f - alpha), k_kairox_lambda_min);

    // TAM 업데이트(ggml_scale_add)는 이 배열을 실행 시점에 그대로 읽는다.
    // 한 스텝 정도 (lambda, 1 - lambda) 가 어긋나도 점수 갱신이 조금 흔들릴 뿐이라 락은 걸지 않는다.
    coeffs[1] = 1.0f - lambda;
    coeffs[0] = lambda;

    lc->dfr_neg_tau = -kairox_tau_load(lambda);

    if (k_kairox_anb_trace) {
        lc->dbg_anb_lambda.push_back(lambda);
        lc->dbg_anb_io_bound.push_back(io_bound ? 1 : 0);
        lc->dbg_anb_reloads.push_back((int32_t) lc->reload_count);
    }
}
```

**(d) `make_anchor()` — 병목 판정 지점**

시그니처를 `std::atomic<int> * dfr_clamp_k, int cap` 에서 `kairox_layer_cache * lc` 로 바꿨다.

```cpp
// lc 를 넘기면 anchor 가 풀리는 시점에 병목을 판정해 balancing 강도를 조절한다(Algorithm 1 Phase 1).
void make_anchor(KairoxWaitType wait_type, kairox_layer_cache * lc = nullptr) {
    ...
    enqueue_io([this, anchor, lc] {
        ...
        if (!lc || k_kairox_dfr_lambda_adapt_rate <= 0.0f) {
            return;
        }

        /**
         * Algorithm 1 line 1: GetSystemBottleneck().
         *
         * anchor 에 매달린 task 는 "reload I/O 가 아직 드레인되는 중에 도착한 sparse compute split"이다.
         * 하나라도 있었다면 계산이 전송을 기다린 것이므로 I/O 병목, 비어 있었다면 전송이 먼저 끝났으므로
         * 남은 시간은 계산이 쓴 것이다 -> CPU 병목.
         */
        const bool io_bound = !to_move.empty();

        if (k_enable_kairox_anb) {
            kairox_anb_feedback(lc, io_bound);
        } else {
            // 논문에 없는 기존 대체 구현: lambda 를 고정한 채 스왑 예산만 조절한다.
            // For simplicity, decrease the maximum load directly when reloading.
            const int cap = lc->cache_shape.n_cached_groups;
            if (cap > 0) {
                int cur = lc->dfr_clamp_k.load();
                int nxt = (int) (cur * (1.0f + (io_bound ? -k_kairox_dfr_lambda_adapt_rate :
                                                            k_kairox_dfr_lambda_adapt_rate)));
                lc->dfr_clamp_k.store(std::clamp(nxt, 1, cap));
            }
        }
    });
}
```

기존 코드의 `to_move.empty()` 분기는 의미를 바꾸지 않고 `io_bound` 로 이름만 뒤집어 옮겼다
(`empty` -> `+rate`, `!empty` -> `-rate` == `io_bound` -> `-rate`).

### 4-2. `ggml/include/ggml.h`, `ggml/src/ggml.c` — `ggml_shifted_step_dyn`

```c
// threshold 를 그래프 빌드 시점에 굽지 않고, 실행 시점에 호스트 메모리에서 읽는 변형.
// threshold 가 스텝마다 바뀌는데 그래프는 재사용되는 경우(KAIROX 의 tau_load)에 쓴다.
// 가리키는 float 은 그래프가 살아있는 동안 유효해야 하며, CPU 에서 읽을 수 있어야 한다.
GGML_API struct ggml_tensor * ggml_shifted_step_dyn(
        struct ggml_context * ctx,
        struct ggml_tensor * a, const float * threshold, bool inplace);
```

```c
struct ggml_tensor * ggml_shifted_step_dyn(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const float         * threshold,
        bool                  inplace) {
    GGML_ASSERT(threshold != NULL);

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    // op_params: [0] = 미사용, [1] = dyn 플래그(1), [2..3] = threshold 를 담은 호스트 메모리 주소
    ggml_set_op_params_f32(result, 0, 0.0f);
    ggml_set_op_params_i32(result, 1, 1);
    memcpy(&result->op_params[2], &threshold, sizeof(threshold));

    result->op     = GGML_OP_SHIFTED_STEP;
    result->src[0] = a;

    return result;
}
```

기존 `ggml_shifted_step` 은 `op_params[0]` 만 쓰고, 텐서 생성 시 `op_params` 는 0 으로 초기화되므로
(`ggml.c:1770` `/*.op_params =*/ { 0 }`) `op_params[1] == 0` 이 정적 경로의 판별자가 된다.

### 4-3. `ggml/src/ggml-cuda/unary.cu` — 커널이 매 실행마다 읽도록

```cpp
float threshold;
memcpy(&threshold, dst->op_params, sizeof(float));

// ggml_shifted_step_dyn: threshold 가 그래프에 상수로 박혀있지 않고 호스트 메모리에 있다.
// 그래프가 재사용되어도 매 실행마다 최신 값을 읽는다.
if (dst->op_params[1] != 0) {
    const float * threshold_src = nullptr;
    memcpy(&threshold_src, &dst->op_params[2], sizeof(threshold_src));
    threshold = *threshold_src;
}
```

### 4-4. `ggml/src/ggml-cuda/ggml-cuda.cu` — 융합 매칭 가드 + 호출부

DFR update 융합(`ggml_cuda_try_kairox_dfr_fusion`)은 `SHIFTED_STEP` 으로 시작하는 서브그래프를 찾는다.
tau 필터의 두 번째 `SHIFTED_STEP` 이 그 머리로 잘못 잡히지 않도록 dyn 노드를 제외한다.

```cpp
// dyn threshold(= tau_load) 를 쓰는 shifted_step 은 DFR update 의 머리가 아니다. 융합 대상에서 제외한다.
if (cgraph->nodes[i]->op == GGML_OP_SHIFTED_STEP && cgraph->nodes[i]->op_params[1] == 0) {
```

호출부:

```cpp
kairox_executor->make_anchor(SingleThreadExecutor::KairoxWaitType::KAIROX_WAIT_AXPY_SPARSE, kairox_lc);
```

### 4-5. `src/llama-graph.cpp` — tau 필터가 dyn 경로를 쓰도록

```cpp
/**
 * tau 는 lambda 에서 유도되는데(Algorithm 1 line 8), ANB 가 켜지면 lambda 가 스텝마다 바뀐다.
 * 그래프는 디코드 중 재사용되므로 빌드 시점 값을 op_params 에 구우면 첫 스텝 값에 고정된다.
 * lc->dfr_neg_tau 를 가리키게 해서 커널이 매 실행마다 최신 -tau 를 읽도록 한다.
 */
// mask 값이 0이면 topk에 포함되지 않도록
ggml_tensor * threshold_mask = ggml_shifted_step_dyn(ctx0, dfr_scores, &lc->dfr_neg_tau, false);
```

### 4-6. `src/llama-kairox.cpp` — 초기화, 로그, 궤적 덤프

```cpp
// ANB 가 켜져 있으면 스왑 예산은 열어두고 lambda 로만 강도를 조절한다
lc->dfr_neg_tau = -kairox_tau_load(k_kairox_lambda_init);
```

시작 시 설정을 한 줄로 남긴다.

```
kairox_cache_manager: ANB on - lambda init 0.670, alpha 0.050, range [0.100, 0.950], tau_load 0.330001 (from lambda)
kairox_cache_manager: ANB off - lambda fixed at 0.670, swap budget adapts at rate 0.050, tau_load 0.330001
```

종료 시 궤적 CSV (`layer,step,lambda,tau_load,io_bound,reloads`) 를 쓴다 (`kairox_dump_anb_trace_csv`).

### 4-7. `test_kairox.sh` — 러너에 노출

`kairox_anb`, `kairox_dfr_lambda_min/max`, `kairox_tau_load`, `kairox_anb_trace` 를 인자로 받아
환경변수로 넘긴다.

```
env CUDA_VISIBLE_DEVICES=0 KAIROX_PARALLEL=1 KAIROX_DFR_LAMBDA_INIT=0.67 KAIROX_DFR_LAMBDA_ADAPT_RATE=0.05 \
    KAIROX_DFR_LAMBDA_MIN=0.10 KAIROX_DFR_LAMBDA_MAX=0.95 KAIROX_ANB=1 KAIROX_TAU_LOAD=0 KAIROX_ANB_TRACE=0 ...
```

---

## 5. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KAIROX_ANB` | `1` | `1`=lambda 적응(논문), `0`=lambda 고정 + 스왑 예산 적응(기존 동작) |
| `KAIROX_DFR_LAMBDA_INIT` | `0.67` | lambda 초기값 (논문 예시는 0.5) |
| `KAIROX_DFR_LAMBDA_ADAPT_RATE` | `0.05` | 논문의 alpha. `0` 이면 적응 자체를 끈다 |
| `KAIROX_DFR_LAMBDA_MIN` / `_MAX` | `0.10` / `0.95` | lambda 상/하한. **논문에 수치 없음** |
| `KAIROX_TAU_EPS` | `1e-6` | tau_load 의 판별 마진 eps |
| `KAIROX_TAU_LOAD` | `0` (미사용) | `>0` 이면 tau 를 lambda 와 무관하게 고정 |
| `KAIROX_ANB_TRACE` | `0` | 레이어별 lambda 궤적 CSV 덤프 |
| `KAIROX_ANB_TRACE_PATH` | `./kairox_anb_trace.csv` | 궤적 CSV 경로 |

---

## 6. 검증

빌드: `bash compile_kairox.sh rel` -> `llama-completion`, `llama-speculative`, `llama-quantize` 전부 통과.

### 6-1. lambda 가 실제로 적응하는가

prosparse-llama-2-7b Q8_0 / RTX 3070 / vb=6 / 512 토큰 생성, `KAIROX_ANB_TRACE=1`.
레이어당 271 스텝이 기록됐다.

| layer | lambda first | min | max | last | io_bound 비율 | 평균 reload |
|---|---|---|---|---|---|---|
| 0 | 0.636 | 0.457 | 0.826 | 0.677 | 51.3% | 46.2 |
| 4 | 0.636 | 0.622 | 0.937 | 0.786 | 51.8% | 21.7 |
| 8 | 0.636 | 0.615 | 0.897 | 0.869 | 52.2% | 23.3 |
| 12 | 0.636 | 0.605 | 0.934 | 0.869 | 52.2% | 24.8 |
| 16 | 0.636 | 0.100 | 0.921 | 0.100 | 22.4% | 16.1 |
| 24 | 0.636 | 0.100 | 0.950 | 0.100 | 22.4% | 15.1 |
| 31 | 0.636 | 0.100 | 0.950 | 0.100 | 21.3% | 15.3 |

얕은 레이어(0-12)는 io_bound 비율이 ~52% 로 **평형점을 찾고** lambda 가 0.68-0.87 근처에서 진동한다.
깊은 레이어(16+)는 io_bound 가 ~22% 라 lambda 가 하한(0.100)까지 내려가 눌린다.

### 6-2. tau 가 커널에 실제로 전달되는가 + lambda=0 가드

`-n 64`, `KAIROX_DUMP_ACTIVATION=1` 로 총 load 수를 센다.

| 설정 | total_loads |
|---|---|
| `lambda=0.00` (가드 적용) | **2,963,664** |
| `lambda=0.00`, `KAIROX_TAU_LOAD=1.000001` (가드 우회 = 옛 동작) | **176** |
| `lambda=0.67`, `ANB=1` | 570,480 |
| `lambda=0.67`, `ANB=0` (기존 동작) | 1,270,240 |

- 2행이 사실상 0 이라는 것은 **`dfr_neg_tau` 값이 정말로 커널까지 전달된다**는 뜻이다
  (포인터가 안 읽혔다면 1행과 2행이 같아야 한다).
- 동시에 `lambda=0` 일 때 tau 가드가 없으면 캐시가 정적으로 굳는다는 것도 확인된다
  — 즉 tau 커밋(`d5a6d17`) 이후 `neuralink` 베이스라인이 조용히 망가져 있었다.

### 6-3. 재현 방법

`bench_anb.sh` 가 아래 조합을 돌리고 카운트 지표 / lambda 궤적 / 레이어별 적중률 세 표를 낸다.
자세한 설명은 README 6-3 장.

```bash
PLATFORM=3070 bash bench_anb.sh full
```

`N=128, BENCH_RUNS=1` 스모크에서 `total_resident` 가 두 조합 모두 28,711,120 으로 **정확히 동일**했다.
캐시 항상-포화 가정(load/evict 개수 강제 일치)이 확인된다.

### 6-4. 주의 — 아직 처리량 비교는 못 했다

이 머신은 RAM 15 GB 에 `--no-mmap` 으로 9 GB 모델을 올리는 구조라, 연속 실행 시 swap 이 터진다
(`pswpout` 316k 페이지). 실제로 같은 설정에서 15.5 t/s 와 0.6 t/s 가 섞여 나왔다.
처리량 비교는 호스트가 한가할 때 캐시를 비우고(`drop_caches`) 다시 재야 한다.

---

## 7. 남은 것 / 알려진 이슈

1. **lambda 하한 포화.** 깊은 레이어에서 lambda 가 `lambda_min` 에 눌린다. 논문 Figure 11 은
   반대로 layer 16/31 에서 lambda 가 0.85 부근에 안정된다고 보고한다. 원인 후보:
   - lambda 를 내리면 교체는 늘지만 동시에 `tau = 1 - lambda` 가 커져 로드를 막는다. 두 효과가
     상쇄되면 피드백이 방향을 되돌릴 신호를 못 받고 하한까지 래칫된다.
   - `lambda_min` 기본값 0.10 이 너무 낮을 수 있다 (논문 미명시). 0.5 부근이 더 맞을 가능성.
2. **tau 필터 입도.** 논문 본문은 뉴런 단위 필터를 설명하고 Algorithm 1 line 12 는 그룹 단위다.
   현재 구현은 그룹 단위 (기존 코드 주석의 의문 그대로).
3. **DFR fusion 미적용 구간.** tau 필터의 `shifted_step` + `mul` 두 노드는 융합되지 않고 별도
   커널로 돈다. n_groups=688 짜리라 비용은 작지만, 정리한다면 dyn threshold 를 받는 융합 경로를 추가하면 된다.
