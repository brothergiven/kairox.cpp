# Wasted Rebalancing Evaluation

## 1. Motivation

KAIROX는 FFN 뉴런을 그룹 단위(16개씩) 로 묶어서 GPU 캐시 교체(rebalancing)를 결정한다.

이 때, 교체 결정은 그룹 단위이지만 실제 연산은 뉴런 단위로 발생하기 때문에 Rebalancing되어 GPU에는 올라갔지만 실제 사용은 되지 않는 "Wasted Rebalancing 현상"이 발생할 수 있다. 

즉, 한 그룹 안 16개 뉴런 중 일부 뉴런만 자주 필요해도 해당 그룹 전체가 Hot으로 분류되어 GPU에 올라가고, 나머지 뉴런들은 GPU 캐시 공간만 차지한 채 사용되지 않을 수도 있다.

## 2. 코드 수정 사항


| 파일 | 역할 |
|---|---|
| `ggml/include/ggml-kairox.hpp` | 계측 On/Off 스위치(`KAIROX_DUMP_ACTIVATION`)와 뉴런별 카운터 필드 5종 정의 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | "이 뉴런이 지금 활성화됐고 + GPU에 상주 중인가"를 판정해서 `dbg_used_since_load`를 세팅 |
| `src/llama-kairox.cpp` | **실제 낭비 판정** — 로드/퇴출이 일어나는 순간 카운터를 갱신, 종료 시 CSV로 덤프 |

### 핵심 로직 — `src/llama-kairox.cpp:79-91`


```cpp
if (k_kairox_dump_activation) { // 계측(KAIROX_DUMP_ACTIVATION)이 켜져 있을 때
    const int evict_base = group_to_evict * group_size; // evict 될 그룹의 시작 인덱스
    const int load_base  = group_to_load * group_size; // load 될 그룹의 시작 인덱스
    for (int k = 0; k < group_size; ++k) { // 그룹 내 뉴런들에 대해
        const int en = evict_base + k; // evict 될 뉴런의 인덱스
        if (!dbg_used_since_load[en]) { // 마지막으로 reload 이후 사용된 적이 없는 뉴런이라면
            ++dbg_wasted_loads[en];  // 로드된 뒤 단 한 번도 안 쓰이고 그대로 쫓겨남 → 낭비 1회 추가
        }
        const int ln = load_base + k; // load 될 뉴런의 인덱스
        dbg_used_since_load[ln] = 0;  // 방금 새로 로드됐으니 "아직 안 씀" 상태로 리셋
        ++dbg_total_loads[ln]; // 해당 뉴런이 reload 된 횟수 증가
    }
}
```

동작 순서는 다음과 같다.

1. **`dbg_used_since_load[n]`** — 뉴런 n마다 1비트 상태를 들고 있다. "마지막으로 로드된 이후
   한 번이라도 사용됐는가"를 나타낸다.
2. 그룹이 **로드(load)** 될 때(87-89행) → 그 그룹에 속한 16개 뉴런 전부 `used_since_load = 0`으로
   리셋하고 `total_loads`를 1씩 증가시킨다 — "이번에 새로 올라왔고, 아직은 안 썼다"는 뜻.
3. 그룹이 **퇴출(evict)** 될 때(83-86행) → 퇴출 직전 `used_since_load`를 검사한다. 여전히 0이면
   (로드된 이후 단 한 번도 켜진 적 없이 쫓겨나는 것이므로) `wasted_loads`를 1 증가시킨다.
4. `used_since_load`가 1로 바뀌는 시점은 이 파일이 아니라 `ggml-cuda.cu`의
   `kairox_dump_activation_counts()`다 — 예측기 점수(`sparse_idx`)가 임계값(0.5) 이상이면서
   동시에 그 뉴런이 지금 GPU에 상주 중일 때만 1로 세팅된다. **활성화됐어도 캐시 밖에 있어서
   CPU 경로로 계산됐다면 세팅되지 않는다** — 재고 싶은 건 "활성화 여부"가 아니라
   "GPU 캐시 슬롯이 쓸모 있었는가"이기 때문이다.
5. 이 판정 함수는 반드시 `kairox_reload_plan()`(위 로직)보다 **먼저** 호출돼야 한다 — 순서가
   바뀌면 "방금 막 로드된 뉴런"이 즉시 상주 상태로 잡혀 착시가 생기는 오프바이원 버그가 된다.


## 3. 실행 방법

### 3-1. 빌드

```bash
bash compile_kairox.sh rel
```

이 때 사용하려는 모델 파일이 VRAM 용량을 초과한다면 양자화를 해준다. (아래 선택 사항 참고)

### 3-2. 계측 켜서 실행

```bash
KAIROX_DUMP_ACTIVATION=1 \
KAIROX_DUMP_ACTIVATION_PATH=./kairox_activation_dump.csv \
  bash test_kairox.sh kairox 3080 kind=completion vb=6 \
  model=/root/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf \
  model_split=/root/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf \
  bench bench_runs=10
```

### 3-3. 결과 확인

```bash
awk -F, 'NR>1 { total+=$6; wasted+=$7 } END { printf "wasted-load rate: %.2f%%\n", wasted/total*100 }' \
  kairox_activation_dump.csv
```

레이어별로 보고 싶으면:

```bash
awk -F, 'NR>1 { total[$1]+=$6; wasted[$1]+=$7 }
         END { for (l in total) printf "layer %2d: %.2f%%\n", l, wasted[l]/total[l]*100 | "sort -n -k2" }' \
  kairox_activation_dump.csv
```

---

<details>
<summary><b>선택 사항 — 다른 환경에서 세팅을 처음부터 다시 잡는 경우</b></summary>

이 실험은 저장소에 미리 세팅된 값(RTX 3080, prosparse-llama-2-7b Q8_0, `vb=6`)으로 바로 재현
가능하도록 준비돼 있다. 다른 GPU/모델로 옮길 때만 아래를 참고한다.

- **GPU 프로파일**: `test_kairox.sh`의 `set_platform_defaults()`에 자신의 GPU 케이스가 없으면
  `gpu_vram=<GiB>`, `threads=<nproc - 1>` 값을 넣어 새 케이스를 추가한다.

- **모델 양자화**: 원본 `prosparse-llama-2-7b.gguf` (F16, 약 16 GiB)가 VRAM에 안 들어가면
  Q8_0으로 재양자화한다. 이때 sparsity 예측기 텐서(`ffn_pred_*`)는 정확도 유지를 위해 반드시
  F16으로 남긴다:

  ```bash
  ./build_rel/bin/llama-quantize --tensor-type ffn_pred=f16 \
    prosparse-llama-2-7b.gguf prosparse-llama-2-7b-Q8_0.gguf Q8_0
  ```

  model-split 파일(`...-sparkinfer-model-split-688.gguf`)은 양자화와 무관하게 그대로 재사용한다.

- **`vb` 값 재탐색**: `vb`(VRAM budget, GiB)는 kairox와 llama.cpp 양쪽에 동일하게 걸리는 GPU
  메모리 예산이다. `vb=0`(무제한)이면 llama.cpp가 메모리를 다 써서 비교가 성립 안 하므로, 총
  VRAM의 60~90% 구간을 짧게 스윕하고 `kairox`의 decode mean이 `llama_cpp`보다 뚜렷하게 큰 지점을
  채택한다.

</details>


## 4. 스윕 스크립트

3절의 수동 실행을 매트릭스로 돌리기 위한 스크립트다. 아티팩트의 처리량 벤치(`bench_models.sh`
-> `test_kairox.sh`)와 같은 드라이버/러너 구조를 계측 쪽에 그대로 옮긴 것이다.

| 스크립트 | 역할 | 대응 |
|---|---|---|
| `dump_activation.sh` | 조합 하나를 실행하는 **러너**. 환경변수로만 설정을 받는다 | `test_kairox.sh` |
| `group_sweep.sh` | group_size 축 하나를 훑는다 (러너를 gs마다 호출) | — |
| `bench_group_sweep.sh` | group_size 스윕 **드라이버**. backend x vb 축을 바깥에 씌운다 | `bench_models.sh` |
| `bench_activation.sh` | activation 프로파일 **드라이버**. vb x backend 축을 훑는다 | `bench_models.sh` |

두 드라이버는 hw 구성을 `test_kairox.sh`의 **3080 프로파일(`gpu_vram=10 GiB`, `threads=12`)로
고정**한다. 다른 GPU로 재려면 러너를 `PLATFORM=` 으로 직접 호출한다.

모델 디렉터리 기본값은 `$HOME/SPIF-GGUF`이고, 없으면 컨테이너 경로 `/root/SPIF-GGUF`를 쓴다.

### 4-1. group_size 스윕 — `bench_group_sweep.sh`

```bash
bash bench_group_sweep.sh simple   # gs 8/16/32 x kairox/vb6, 3조합 (동작 확인)
bash bench_group_sweep.sh full     # 기본 매트릭스 (생략 시 full)

# 축을 좁혀서 gs 경향만 먼저 보기 (7조합, 약 10분)
BENCH_RUNS=2 BACKENDS="kairox" VBS="6" bash bench_group_sweep.sh full
```

| 변수 | 기본값 (`full` / `simple`) | 설명 |
|---|---|---|
| `SIZES` | `"2 4 8 16 32 64 128"` / `"8 16 32"` | group_size 목록. `n_ff`(11008)로 나누어떨어져야 한다 |
| `VBS` | `"5 6 7"` / `"6"` | VRAM budget(GiB) 목록. 3080 고정이라 10 미만이어야 한다 |
| `BACKENDS` | `"kairox neuralink"` / `"kairox"` | lambda 프로파일 |
| `BENCH_RUNS` | `5` / `2` | 조합당 프롬프트 개수 |
| `N` | `512` | 프롬프트당 생성 토큰 수 |
| `PROMPT_FILE` | `./prompts.txt` | 한 줄 = 프롬프트 하나인 집합 파일 |
| `MODEL_DIR` / `MODEL` | `$HOME/SPIF-GGUF` / `prosparse-llama-2-7b-Q8_0.gguf` | |
| `OUT_DIR` | `./group_sweep_logs` | 결과 디렉터리 |
| `REGROUP` | `1` | 없는 model-split을 `regroup_model_split.py`로 생성. `0`이면 그 gs를 건너뜀 |
| `REBUILD` | `0` | `1`이면 `build_rel`을 지우고 새로 빌드 |
| `FORCE` | `0` | `1`이면 이미 있는 CSV도 다시 측정 |

출력:

```text
group_sweep_logs/
  kairox__vb6/gs16.csv                                        # 뉴런별 raw (gs마다 하나)
  group_sweep__kairox__3080__completion__<model>__vb6.log      # 조합별 실행 로그
  group_sweep_summary.csv                                      # 조합별 집계
```

콘솔 마지막에 gs를 행, `(backend, vb)`를 열로 둔 비교표 3장(`hit/activation`, `hit/resident`,
`wasted/total`)이 나온다.

### 4-2. activation 프로파일 — `bench_activation.sh`

group_size는 고정하고 vb(메모리 압박)와 backend 축을 보는 쪽이다.

```bash
bash bench_activation.sh simple
bash bench_activation.sh full
```

`SIZES` 대신 `GROUP_SIZES`(기본 `"16"` = split-688)를 쓰고, `REPEAT`(기본 1)로 같은 조합을 여러
번 반복할 수 있다는 점만 다르다. `VBS` 기본값은 `"4 5 6 7 8"`, 그 외 변수는 4-1과 같다.

출력은 `activation_logs/` 아래에 `bench_models.sh`의 로그 이름 규칙을 따라 쌓인다:

```text
<benchmark_group>__<backend>__3080__completion__<model>__gs<N>__vb<M>.csv / .log
activation_summary.csv     # decode t/s 포함
```

### 4-3. 러너 직접 호출 — `dump_activation.sh`

```bash
VB=6 BACKEND=kairox \
  MODEL_SPLIT=$HOME/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split-2752.gguf \
  OUT=/tmp/gs4.csv bash dump_activation.sh
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PLATFORM` | `3080` | `3080`(10 GiB/12), `3080ti`(12/12), `4090`(24/16), `3070`(8/7) |
| `VB` / `THREADS` | 프로파일 기본값 | VRAM budget(GiB) / CPU 스레드 |
| `BACKEND` | `kairox` | `kairox`=lambda 0.67/adapt 0.05, `neuralink`=0.00/0.00 |
| `LAMBDA_INIT` / `LAMBDA_ADAPT` | BACKEND 값 | 개별 덮어쓰기 |
| `MODEL_SPLIT` | `...-split-688.gguf` | **group_size는 이 파일이 결정한다** |
| `PROMPT_FILE` / `BENCH_RUNS` | `./prompts.txt` / `5` | 프롬프트 집합과 그중 몇 개를 돌릴지 |
| `PROMPT` | — | 문자열을 주면 그 프롬프트 하나만 (`-p`, `PROMPT_FILE` 무시) |
| `N` / `CTX` / `SEED` | `512` / `1024` / `42` | |
| `IGNORE_EOS` | `1` | `--ignore-eos`로 N 토큰을 강제 생성 |
| `OUT` / `LOG` / `SUMMARY` | `./kairox_activation.csv` / — / `1` | CSV 경로 / 실행 로그 / 요약표 출력 |

프롬프트는 `--bench-prompt-file`로 넘어가고, 계측 카운터는 런 사이에 리셋되지 않으므로 CSV는
**여러 프롬프트에 걸친 합계**가 된다. 워밍업 런도 카운터에 누적되기 때문에 `--bench-warmup 0`을
쓴다.

### 4-4. 재개와 재측정

이미 CSV가 있는 조합은 건너뛴다. 축을 좁혀 돌린 뒤 넓혀서 다시 돌리면 새 조합만 측정하고,
요약표는 매번 누적된 전체를 기준으로 다시 그린다.

```bash
# 1단계: gs 축만 (~10분)
BENCH_RUNS=2 BACKENDS="kairox" VBS="6" bash bench_group_sweep.sh full
# 2단계: vb 축 확장 — 1단계 결과는 재사용된다
BENCH_RUNS=2 BACKENDS="kairox" VBS="5 6 7" SIZES="8 16 32" bash bench_group_sweep.sh full
# 3단계: 확정된 조합만 표본을 늘려 재측정
FORCE=1 BENCH_RUNS=10 VBS="6" SIZES="8 16 32" bash bench_group_sweep.sh full
```

표만 다시 그리고 싶을 때도 같은 명령을 다시 돌리면 된다 (전부 skip되고 CSV에서 표만 생성).

### 4-5. 주의

- **실행 시간**은 `조합 수 x BENCH_RUNS x N`에 비례한다. `bench_group_sweep.sh full` 기본값은
  42조합이라 몇 시간 단위다. 축이나 `BENCH_RUNS`를 먼저 줄여서 경향을 본다.
- 전체 출력은 로그 파일에 남고, 콘솔에는 진행을 알 수 있는 줄(gs 시작, 런별 t/s, `decode mean`,
  CSV 기록 완료, warning/error)만 실시간으로 흐른다. 런 하나에 수십 초~수 분 걸리므로 그
  간격만큼 조용한 것은 정상이다. 로그를 통째로 보려면:

  ```bash
  tail -f group_sweep_logs/group_sweep__kairox__3080__completion__*__vb6.log
  ```

- CSV는 프로세스가 **정상 종료할 때** 소멸자에서 쓰인다. `Ctrl+C`로 끊으면 그 조합은 파일이
  남지 않는다.
- `group_size=1`은 `group_identity`가 11008^2 x 4 = 약 462 MiB라 VRAM을 크게 먹는다. 기본
  `SIZES`에서 뺐으니 필요하면 직접 지정한다.
- 스크립트가 도는 동안 스크립트 파일을 편집하지 않는다. bash는 실행하면서 파일을 이어 읽기
  때문에 엉뚱한 지점으로 튈 수 있다.

## 5. 실험 결과 (요약)


| GPU | 총 VRAM | vb | 전체 로드 | 낭비된 로드 | **낭비율** |
|---|---|---|---|---|---|
| RTX 3070 | 8 GiB  | 6 | 8,806,112 | 4,576,507 | **51.97%** |
| RTX 3080 | 10 GiB | 6 | (측정)     | (측정)     | **54.51%** |


## 6. Adaptive Neuron Balancer (ANB) — 논문 Algorithm 1 Phase 1

논문 Algorithm 1 중 Phase 2(TAM 갱신 + tau_load 필터 + top-K)는 이미 구현돼 있었지만,
Phase 1(병목 피드백으로 lambda 를 조절하는 부분)은 없었다. 대신 lambda 를 0.67 로 고정하고
스왑 예산(`dfr_clamp_k`)을 대신 조절하는, 논문에 없는 경로가 들어가 있었다.

### 6-1. 구현

| 위치 | 내용 |
|---|---|
| `ggml/include/ggml-kairox.hpp` | `kairox_anb_feedback()` — Algorithm 1 line 1-6. `kairox_tau_load()` — line 8 |
| `ggml/include/ggml-kairox.hpp` | `make_anchor()` 가 병목을 판정해 lambda(ANB on) 또는 스왑 예산(ANB off)을 조절 |
| `ggml/src/ggml.c`, `ggml/src/ggml-cuda/unary.cu` | `ggml_shifted_step_dyn()` — threshold 를 실행 시점에 호스트 메모리에서 읽는 변형 |
| `src/llama-graph.cpp:1329` | tau 필터가 `ggml_shifted_step_dyn` 을 쓰도록 변경 |
| `src/llama-kairox.cpp` | lambda/tau 초기화, ANB 궤적 CSV 덤프 |

**병목 판정.** 논문의 `GetSystemBottleneck()` 은 "파이프라인 stall 을 관찰한다"고만 돼 있다.
여기서는 이미 있던 anchor 신호를 쓴다 — `SingleThreadExecutor` 의 anchor 에 매달린 task 는
"reload I/O 가 드레인되는 중에 도착한 sparse compute split" 이므로, 하나라도 있으면 계산이
전송을 기다린 것(IO_BOUND)이고 비어 있으면 전송이 먼저 끝난 것(CPU_BOUND)이다.

**tau 를 상수로 구우면 안 되는 이유.** `tau_load = (1 - lambda) + eps` 인데 lambda 가 스텝마다
바뀐다. llama.cpp 는 디코드 중 그래프를 재사용하므로(실측 512 토큰 중 `graphs reused = 269`)
빌드 시점 값을 `op_params` 에 구우면 첫 스텝 값에 그대로 고정된다. 그래서 커널이 매 실행마다
`kairox_layer_cache::dfr_neg_tau` 를 읽도록 `ggml_shifted_step_dyn` 을 추가했다.

**lambda = 0 예외.** `neuralink` 프로파일은 lambda=0 이다. 이때 S = A <= 1 인데
tau = (1-0) + eps = 1+eps 라 어떤 그룹도 필터를 통과하지 못해 캐시가 정적으로 굳는다.
관성이 없으면 one-hit wonder 라는 개념 자체가 없으므로 lambda <= 0 이면 필터를 끈다.

### 6-2. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KAIROX_ANB` | `1` | `1`=lambda 적응(논문), `0`=lambda 고정 + 스왑 예산 적응(기존 동작) |
| `KAIROX_DFR_LAMBDA_INIT` | `0.67` | lambda 초기값 (논문 예시는 0.5) |
| `KAIROX_DFR_LAMBDA_ADAPT_RATE` | `0.05` | 논문의 alpha. `0` 이면 적응 자체를 끈다 |
| `KAIROX_DFR_LAMBDA_MIN` / `_MAX` | `0.10` / `0.95` | lambda 상/하한. 논문에 수치 없음 |
| `KAIROX_TAU_EPS` | `1e-6` | tau_load 의 판별 마진 eps |
| `KAIROX_TAU_LOAD` | `0` (미사용) | `>0` 이면 tau 를 lambda 와 무관하게 고정 (필터 강도 스윕용) |
| `KAIROX_ANB_TRACE` | `0` | 레이어별 lambda 궤적을 CSV로 덤프 (논문 Figure 11 대응) |
| `KAIROX_ANB_TRACE_PATH` | `./kairox_anb_trace.csv` | 궤적 CSV 경로 |

```bash
KAIROX_ANB=1 KAIROX_ANB_TRACE=1 KAIROX_ANB_TRACE_PATH=./anb.csv \
  bash test_kairox.sh kairox 3070 kind=completion vb=6
```

궤적 CSV 컬럼: `layer,step,lambda,tau_load,io_bound,reloads`.

### 6-3. ANB 스윕 — `bench_anb.sh`

`bench_activation.sh` 와 같은 드라이버/러너 구조다. 처리량이 아니라 **카운트 지표**
(hit / activation / resident / loads / wasted)를 재는데, 처리량 노이즈가 ±45% 라
20~30% 미만의 효과는 측정 자체가 되지 않기 때문이다. 카운트 지표는 같은 프롬프트 집합에
`--ignore-eos` 를 걸면 재현된다.

```
bench_models.sh      -> test_kairox.sh       (throughput)
bench_activation.sh  -> dump_activation.sh   (activation/hit/resident)
bench_group_sweep.sh -> group_sweep.sh       (group_size 스윕)
bench_anb.sh         -> dump_activation.sh   (ANB 설정 스윕)
```

```bash
# 동작 확인 (2조합)
PLATFORM=3070 N=128 BENCH_RUNS=1 bash bench_anb.sh simple

# 전체 (4조합)
PLATFORM=3070 bash bench_anb.sh full
```

기본 조합:

| 이름 | ANB | lambda_init | alpha | lambda 범위 | 의미 |
|---|---|---|---|---|---|
| `anb_off` | 0 | 0.67 | 0.05 | — | 기존 동작 (lambda 고정, 스왑 예산 적응) |
| `anb_frozen` | 1 | 0.67 | 0.05 | [0.67, 0.67] | 널 컨트롤 — 피드백은 돌지만 lambda 가 못 움직인다 |
| `anb_min010` | 1 | 0.67 | 0.05 | [0.10, 0.95] | 논문 Phase 1, 하한 기본값 |
| `anb_min050` | 1 | 0.67 | 0.05 | [0.50, 0.95] | 논문 Phase 1, tau <= 0.5 로 제한 |

`CONFIGS` 로 덮어쓸 수 있고 `anb_static`(적응 전혀 없음), `anb_paper`(논문 예시 초기값 0.5)도 있다.

주요 환경변수 — `VBS`(기본 `"6"`), `CONFIGS`, `PLATFORM`, `BENCH_RUNS`, `N`(**32 이상**),
`REPEAT`, `OUT_DIR`(기본 `./anb_logs`), `FORCE`, `COOLDOWN`, `DROP_CACHES`.

출력은 세 갈래다.

1. **카운트 지표** (`anb_summary.csv`) — 조합별 hit/res, hit/act, 낭비율, total_loads, resident
2. **lambda 궤적 요약** (`anb_lambda_summary.csv`) — 수렴 후(마지막 1/4 구간) 얕은 층/깊은 층
   평균 lambda, io_bound 비율, 스텝당 reload. 논문 Figure 11 과 대조하는 용도다.
3. **레이어별 hit/act 표** — 조합을 나란히 놓고 본다

무엇을 확인해야 하는지:

- `resident` 열은 **정책과 무관하게 `n_cached_neurons x 총 토큰 수` 로 고정**이어야 한다
  (캐시는 항상 꽉 차 있고 load/evict 개수가 강제로 같다). 조합마다 다르면 그 가정부터 깨진 것이다.
- `anb_min010` 에서 깊은 층 lambda 가 하한에 눌리는지 본다. lambda 를 내리면 교체가 늘어야 하는데
  `tau = (1 - lambda)` 가 같이 커져 로드를 막는 상쇄가 있다.
- `anb_min050` 에서 그 상쇄가 풀려 lambda 가 하한을 떠나는지, 적중률이 회복되는지 본다.

> **주의.** 호스트 RAM 이 모델보다 빠듯하면(`--no-mmap` 이라 9 GB 모델이 통째로 올라간다)
> 연속 실행이 swap 으로 무너진다. `DROP_CACHES=1 COOLDOWN=20` 을 쓰고, 다른 작업이 도는
> 중에는 처리량 측정을 하지 말 것.


## 7. 실험 기록

측정 조건은 따로 적지 않는 한 공통이다 — prosparse-llama-2-7b Q8_0, RTX 3070 8 GiB, vb=6 GiB,
프롬프트 2개 x 512토큰(`--ignore-eos` 로 토큰 수 고정), 상주 슬롯-토큰 182.5M(전 조건 동일).

> **표본 주의.** 프롬프트 2개는 경향 확인용이다. 적중률은 카운트 기반이라 안정적이지만
> 시간은 이 박스에서 노이즈가 ±45% 다. 1~2%p 차이는 결론에 쓰지 않는다.

### 7-1. τ_load 필터 — 논문 설명과 반대로 동작한다

논문은 τ_load = (1−λ)+ε 필터가 one-hit wonder 를 걸러 reload latency 를 1.8–2.2× 줄인다고
설명한다. 실측은 반대다. g=16 고정, τ 만 바꿔 스윕:

| τ_load | 적중률 (clamp 무제한) | 로드 | 적중률 (clamp 활성) | 로드 |
|---|---|---|---|---|
| ~0 (off) | 68.50% | 13,040,704 | 69.54% | 13,093,024 |
| **0.05** | **68.83%** | 13,585,120 | **70.82%** | 13,732,736 |
| 0.10 | 68.76% | 13,475,424 | 70.26% | 13,255,456 |
| 0.20 | 66.75% | 14,604,960 | 70.14% | 14,545,568 |
| **0.33 (논문값)** | **60.97%** | **19,358,880** | **65.07%** | 15,818,912 |
| 0.50 | 56.48% | 11,273,056 | 55.99% | 9,196,048 |

곡선에 봉우리가 없다. τ≈0–0.10 이 평평한 고원이고 0.20 부터 무너진다. **필터를 세게 걸수록
로드가 늘어난다**(τ=0.33 에서 τ≈0 대비 +49%)는 것이 결정적이다 — τ 가 막던 것은 일회성
뉴런이 아니라 오래 상주할 좋은 그룹이었고, 그것들이 배제되니 캐시가 자리를 못 잡고 같은
슬롯을 반복 교체했다.

두 계열은 **모양이 같다**(서로 다른 조건에서 재현). 다만 겹치지는 않는다 — clamp 활성 쪽이
6점 중 5점에서 높고 격차는 τ=0.33 에서 4.1%p 로 가장 크다. clamp 가 τ 의 해악을 부분적으로
완충하는 것으로 보인다.

**원인 후보.** τ 는 그룹 **평균** 활성률(0–1)과 비교된다. g=16 에서 τ=0.33 은 "16개 중 5~6개
동시 활성"을 요구하는데, 코액티베이션 응집도가 C(16)=27.5%(16개 중 4.4개)라 정상적인 hot
그룹조차 통과하지 못한다.

### 7-2. 그룹 입도 — 고울수록 적중률이 오른다

용량을 고정한 채 group_size 만 바꿨다. 세 조건은 τ 와 스왑 예산 설정만 다르다.

| g | n_group | 기본 설정 | 스로틀 제거 | **τ+스로틀 제거** | 시간(τ+스로틀 제거) |
|---|---|---|---|---|---|
| 2 | 5504 | 63.19% | 75.05% | **80.06%** | 490.0s |
| 4 | 2752 | 66.46% | 64.65% | **76.01%** | 239.3s |
| 8 | 1376 | 66.17% | 65.48% | **72.35%** | 140.1s |
| 16 | 688 | 64.16% | 62.49% | **68.39%** | 128.6s |
| 32 | 344 | 59.76% | 62.37% | 68.81% | 123.1s |
| 64 | 172 | 60.63% | 61.25% | 66.36% | 124.7s |
| 128 | 86 | 58.75% | 58.93% | 65.63% | 125.0s |

격리 조건에서 **g=128 → 2 로 −14.4%p, 사실상 단조**다(g=32 만 0.42%p 역전, 노이즈 범위).
미스 기준으로는 34.37% → 19.94%, **미스가 42% 줄어든다.** 배포값 g=16 대비로는 +11.67%p.

기본/스로틀 제거 조건에서 보이던 g=2 함몰과 g=4 딥은 각각 스왑 예산과 τ 가 만든 인공물이었다.

**입도 효과는 조건에 무관하다.** g=8→128 구간 기울기가 −7.42 / −6.55 / −6.72 %p 로 모인다.
절편은 6%p 넘게 흩어지는데 기울기는 같다 — τ 는 곡선을 위아래로 옮길 뿐 기울기를 바꾸지 않는다.

**그런데 시간은 정반대다.** 적중률 최고점(g=2, 80.06%)이 시간 최악(490s, g=128 의 4배)이다.
전송이 그룹당 `cudaMemcpyAsync` 3회로 **그룹 크기와 무관**하므로, g=2 는 같은 뉴런 수를 옮기는 데
g=16 의 8배 호출을 쓴다. 이 표의 적중률은 최적화 결과가 아니라 **아직 못 쓰고 있는 잔고**다.

### 7-3. 스왑 예산 `dfr_clamp_k` — g=2 에서만 작동한다

`llama-kairox.cpp` 는 정책이 결정한 교체 목록을 `dfr_clamp_k` 개까지만 실행하고 나머지를 버린다.
논문에 없는 장치다(주석: "For simplicity, decrease the maximum load directly when reloading").

| g | 로드 (clamp 활성) | 로드 (무제한) | 배율 | Δ적중률 |
|---|---|---|---|---|
| 2 | 8,153,748 | 20,443,748 | **2.51x** | **+11.86p** |
| 4 | 12,637,408 | 19,931,528 | 1.58x | −1.81p |
| 8 | 14,035,992 | 15,882,728 | 1.13x | −0.70p |
| 16 | 16,098,832 | 18,925,344 | 1.18x | −1.67p |
| 32 | 14,628,896 | 15,396,928 | 1.05x | +2.61p |
| 64 | 16,254,848 | 14,043,200 | **0.86x** | +0.62p |
| 128 | 12,086,144 | 13,320,704 | 1.10x | +0.18p |

g=2 에서는 정책 요구의 40% 만 실행하고 있었다. 그런데 g≥4 에서는 배율이 1.05~1.18 배뿐이고,
**g=64 에서는 0.86 배**다 — 스로틀을 없앴는데 로드가 더 적으니 애초에 물고 있지 않았다.

자기 안정화 구조 때문이다. 잘라내면 I/O 가 줄어 `anchor->pending` 이 비고, 그러면 +5% 로 다시
자란다. 그래서 "가끔 아주 살짝 무는" 지점 근처를 맴돈다. 전송이 호출 고정비에 묶여 있어
g=2 만 항상 I/O 병목으로 판정되고 곱셈적으로 바닥까지 내려간다.

### 7-4. 비교군 `neuralink` 가 죽어 있었다

τ 커밋(`d5a6d17`) 이후 λ=0 인 `neuralink` 는 τ=(1−0)+ε=1+ε 가 되어 어떤 그룹도 통과하지
못한다(점수 S 는 정의상 1 을 넘을 수 없다). 캐시가 정적으로 굳는다.

| 설정 | 총 로드 |
|---|---|
| λ=0, 가드 우회 (옛 동작) | **176** |
| λ=0, 가드 적용 | **2,963,664** |

이 A/B 는 동시에 τ 값이 실제로 커널까지 전달된다는 증거이기도 하다(포인터가 안 읽혔다면
두 값이 같아야 한다). **해당 커밋 이후 측정한 neuralink 수치는 전부 무효다.**

### 7-5. 진행 중 / 미실행

- **전송 병합(gather).** `exp/group-granularity` 의 `kairox_gather_reload()` 를 이 브랜치로
  가져왔다(`KAIROX_GATHER=1`). g=4/8/16 을 전송 경로만 바꿔 재는 중이다. 판정 기준: 적중률은
  양쪽이 같아야 하고(정책 불변), 승부는 시간에서 난다.
- **여러 모델.** `gs_multi_model.sh` 로 돌린다. 배포 split 의 n_group 이 전 모델에서 1024
  이하이므로(논문 8장의 argsort 제약), g=16 은 워크로드 최적값이 아니라 7B급에서 그 제약이
  허용하는 가장 고운 값일 가능성이 높다. 자세한 것은 [CROSS_MODEL_RUNBOOK.md](CROSS_MODEL_RUNBOOK.md).
- **ANB 자체의 처리량 이득.** 미판정. 호스트 스왑 오염으로 무산됐다.

---

## 8. 스크립트 색인

드라이버는 여러 조합을 돌리고, 러너는 조합 하나를 실행한다.

| 드라이버 | 러너 | 무엇을 재나 |
|---|---|---|
| `bench_models.sh` | `test_kairox.sh` | 처리량 (AE 경로) |
| `bench_activation.sh` | `dump_activation.sh` | activation / hit / resident |
| `bench_group_sweep.sh` | `group_sweep.sh` | group_size 스윕 |
| `bench_anb.sh` | `dump_activation.sh` | ANB 설정 스윕 |
| `gs_multi_model.sh` | `dump_activation.sh` | 여러 모델 x group_size |

```bash
# 빌드
bash compile_kairox.sh rel

# ANB 설정 비교 (anb_off / anb_frozen / anb_min010 / anb_min050)
PLATFORM=3070 bash bench_anb.sh full

# group_size 스윕 — 정책 격리 조건
ANB=1 LAMBDA_MIN=0.67 LAMBDA_MAX=0.67 TAU_LOAD=0.0001 \
  SIZES="2 4 8 16 32 64 128" OUT_DIR=./dumps_pure bash group_sweep.sh

# τ 스윕 (g=16 고정)
for tau in 0.0001 0.05 0.10 0.20 0.33 0.50; do
  ANB=1 LAMBDA_MIN=0.67 LAMBDA_MAX=0.67 TAU_LOAD=$tau \
    OUT=./tau_$tau.csv bash dump_activation.sh
done

# 전송 병합 A/B
KAIROX_GATHER=1 VB=6 bash dump_activation.sh

# 여러 모델
bash gs_multi_model.sh
MODELS="opt-6.7b opt-13b" SIZES="4 16 64" bash gs_multi_model.sh
```

### 환경변수 한눈에

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KAIROX_PARALLEL` | `0` | **필수.** 0 이면 재배치 자체가 꺼져 캐시가 정적이 된다 |
| `KAIROX_ANB` | `1` | 1=λ 적응(논문 Phase 1), 0=λ 고정 + 스왑 예산 적응(기존) |
| `KAIROX_DFR_LAMBDA_INIT` | `0.67` | λ 초기값 (논문 예시는 0.5) |
| `KAIROX_DFR_LAMBDA_ADAPT_RATE` | `0.05` | 논문의 α. 0 이면 적응 자체를 끈다 |
| `KAIROX_DFR_LAMBDA_MIN` / `_MAX` | `0.10` / `0.95` | λ 상/하한. 논문에 수치 없음 |
| `KAIROX_TAU_EPS` | `1e-6` | τ_load 의 판별 마진 ε |
| `KAIROX_TAU_LOAD` | `0` (미사용) | >0 이면 τ 를 λ 와 무관하게 고정 |
| `KAIROX_ANB_TRACE` / `_PATH` | `0` / `./kairox_anb_trace.csv` | λ 궤적 CSV (논문 Figure 11 대응) |
| `KAIROX_GATHER` | `0` | 1 이면 전송을 pinned staging 으로 묶는다 |
| `KAIROX_GATHER_BUDGET_MIB` | `64` | gather staging 버퍼 예산 |
| `KAIROX_GATHER_VERIFY` | `0` | gather 경로 바이트 단위 검증. 매우 느리다 |
| `KAIROX_RELOAD_WINDOW` | `4` | 개별 전송 경로의 동기화 주기 |
| `KAIROX_DUMP_ACTIVATION` / `_PATH` | `0` / `./kairox_activation.csv` | 뉴런별 카운터 CSV |


---

# KAIROX Artifact Evaluation

For artifact evaluation of KAIROX, see [ARTIFACTS_EVALUATION.md](ARTIFACTS_EVALUATION.md).

----

# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
