# Wasted Rebalancing Evaluation

## 1. Motivation

KAIROX는 FFN 뉴런을 그룹 단위(16개씩)로 묶어서 GPU 캐시 교체(rebalancing)를 결정한다.

교체 결정은 그룹 단위이지만 실제 연산은 뉴런 단위로 발생한다. 따라서 한 그룹 안 16개 뉴런
중 일부만 필요해도 그룹 전체가 GPU에 올라가고, 나머지는 캐시 공간만 차지한 채 쓰이지 않는
"Wasted Rebalancing"이 발생할 수 있다.

이를 계측한 결과(4절), 기준 설정에서 **리밸런싱 로드의 45.14%가 한 번도 쓰이지 않고
축출된다.** 결정 단위를 뉴런 단위까지 좁히면 이 값은 19.73%까지 떨어지고 실제 PCIe 전송량은
4.5배 줄어든다(5-2절).

다만 원인은 처음 예상과 달랐다. **캐시 적중률은 결정 단위와 무관하게 일정하다**(64배 범위에서
59.6~62.3%). 그룹을 잘게 쪼갠다고 예측이 더 맞는 것이 아니라, **거친 그룹이 같은 적중률을
얻기 위해 불필요한 데이터를 훨씬 많이 옮기고 있는 것**이다.

이 결과는 문제를 독립적인 두 가지로 분리한다(5-5절).

- **전송 구조** — `group_size`가 DFR 점수 계산 단위이자 evict/load 결정 단위이면서 동시에
  DMA 전송 단위를 겸한다. 그래서 결정을 세밀하게 하려면 전송이 잘게 쪼개지고, 전송을
  키우면 불필요한 뉴런까지 딸려온다. 측정된 decode 시간의 지배 요인은 PCIe 대역폭이 아니라
  **전송 호출 횟수**였다(5-3절).
- **예측 정책** — 뉴런 단위로 완벽히 골라도 적중률이 61%에서 멈춘다. 같은 용량에 활성화
  빈도 상위 N개를 고정 배치하면 70~85%에 도달한다는 점을 감안하면, EMA 기반 DFR 점수
  자체가 천장이다.

이 문서는 전자를 다룬다. gather 버퍼로 결정 단위와 전송 단위를 분리하는 해법을 구현해
측정했다(5-6, 5-7절). 결과를 한 줄로 요약하면:

> **gather는 처리량 천장을 올리지 않는다. fine-grained 결정을 감당 가능하게 만든다.**
> gs=4에서 baseline 최선(gs=64)과 같은 17.2 tok/s를 내면서 PCIe 트래픽은 3.9배,
> 낭비 적재는 1.4배 적다. gather를 켜면 gs=4/16/64가 전부 17.2~17.7 tok/s로 수렴한다 —
> 전송량이 3.9배 차이나는데도 그렇다.

후자(예측 정책)는 6장에서 다룬다. 요약하면 **논문의 Adaptive Neuron Balancer(§7)가
구현돼 있지 않고**, 아티팩트가 고정한 λ=0.67 은 논문 자신의 수렴값 0.85 보다도 나쁘다.
λ 만 0.99 로 바꾸면 처리량 +24.8%, 적중률 +4.6p, PCIe 트래픽 1/19 이다.

설정·전송·정책 세 가지를 합치면 **13.78 → 25.87 tok/s (1.88배)** 이며, 새 알고리즘은
하나도 없다.

## 2. 코드 수정 사항


| 파일 | 역할 |
|---|---|
| `ggml/include/ggml-kairox.hpp` | 계측 On/Off 스위치(`KAIROX_DUMP_ACTIVATION`)와 뉴런별 카운터 필드 6종 정의 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | "이 뉴런이 지금 활성화됐고 + GPU에 상주 중인가"를 판정해서 `dbg_used_since_load` / `dbg_hit_count`를 세팅 |
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
   `kairox_dump_residency_counts()`다 — 예측기 점수(`sparse_idx`)가 임계값(0.5) 이상이면서
   동시에 그 뉴런이 지금 GPU에 상주 중일 때만 1로 세팅된다. **활성화됐어도 캐시 밖에 있어서
   CPU 경로로 계산됐다면 세팅되지 않는다** — 재고 싶은 건 "활성화 여부"가 아니라
   "GPU 캐시 슬롯이 쓸모 있었는가"이기 때문이다.
5. 이 판정 함수는 반드시 `kairox_reload_plan()`(위 로직)보다 **나중에** 호출돼야 한다.
   스왑이 끝난 뒤의 `neuron_mask`가 곧 이 레이어의 sparse matmul이 실제로 읽게 될 캐시
   상태이기 때문이다. 스왑 전 마스크로 채점하면 "지금 필요해서 방금 로드한 뉴런"의 사용이
   통째로 누락되어 `wasted_loads`가 구조적으로 부풀려진다.
6. 누적 적중 카운터(`dbg_hit_count`)와 낭비 판정용 플래그(`dbg_used_since_load`)는 반드시
   분리해야 한다. 후자는 로드 시 0으로 리셋되므로, 하나로 겸하면 지표 1·2가 과소집계된다.
7. `total_loads == 0`인 뉴런은 리밸런싱으로 로드된 적이 없는 초기 상주분이므로 낭비 판정에서
   제외한다. 초기 배치는 `[0, n_cached_neurons)` 고정으로 DFR의 결정이 아니기 때문이다.


## 3. 실행 방법

### 3-1. 빌드

```bash
bash compile_kairox.sh rel
```

이 때 사용하려는 모델 파일이 VRAM 용량을 초과한다면 양자화를 해준다. (아래 선택 사항 참고)

### 3-2. 계측 켜서 실행

```bash
bash dump_activation.sh
```

`PLATFORM`(3070/3080), `VB`, `N`, `OUT`, `MODEL_SPLIT`, `IGNORE_EOS` 환경변수로 조정한다.
비교 실험에서는 런마다 생성 길이가 달라지지 않도록 `IGNORE_EOS=1`을 쓴다.

### 3-3. group_size 스윕

model-split GGUF의 `ffn_group_size`만 바꿔 재작성한 뒤 스윕한다.

```bash
for gs in 1 2 4 8 32 64; do
  python3 regroup_model_split.py ~/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf $gs
done

SIZES="1 2 4 8 16 32 64" bash group_sweep.sh
```

### 3-4. PCIe 마이크로벤치마크

```bash
nvcc -O3 -std=c++17 -o bench_pcie bench_pcie.cu && ./bench_pcie
```

### 3-5. gather A/B

```bash
# 1단계(정확성) + 2단계(처리량) 전부
bash gather_ab.sh

# 처리량만, group_size 를 바꿔서
GS=4 ONLY=2 bash gather_ab.sh

# 바이트 단위 검증만 따로 (매우 느림)
KAIROX_GATHER=1 KAIROX_GATHER_VERIFY=1 GATHER=1 DUMP=0 N=32 bash dump_activation.sh
```

주요 노브:

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `KAIROX_GATHER` | 0 | 1이면 gather/scatter 경로 |
| `KAIROX_GATHER_BUDGET_MIB` | 64 | staging 버퍼 예산, 초과 시 청크 분할 |
| `KAIROX_GATHER_VERIFY` | 0 | scatter 결과를 되읽어 원본과 memcmp |
| `KAIROX_RELOAD_WINDOW` | 4 | 기존 경로의 동기화 주기 |

`dump_activation.sh`는 `GATHER` / `RELOAD_WINDOW` / `DUMP` 로 이들을 받는다.
**처리량을 비교할 때는 `DUMP=0`이 필수다** — 계측이 레이어마다 D2H 복사와
`cudaStreamSynchronize`를 추가한다.

### 3-6. 결과 확인

```bash
awk -F, 'NR>1 { act+=$3; res+=$4; hit+=$5; tot+=$6; wst+=$7 }
  END { printf "hit/resident   : %.2f%%\n", hit/res*100
        printf "hit/activation : %.2f%%\n", hit/act*100
        printf "wasted/total   : %.2f%%\n", wst/tot*100 }' kairox_activation.csv
```

레이어별로 볼 때는 `gpu_only` 레이어가 `total_loads = 0`이므로 0 나눗셈을 걸러야 한다
(gawk는 0으로 나누면 중단된다):

```bash
awk -F, 'NR>1 { t[$1]+=$6; w[$1]+=$7 }
  END { for (l in t) if (t[l] > 0) printf "layer %2d: %6.2f%%\n", l, w[l]/t[l]*100 }' \
  kairox_activation.csv | sort -n -k2
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


## 4. 측정 결과

측정 조건: RTX 3070 (8 GiB), `vb=6`, prosparse-llama-2-7b Q8_0, 단일 프롬프트,
`--ignore-eos`로 생성 길이를 511 decode 토큰으로 고정(총 564 토큰 스텝).
`KAIROX_PARALLEL=1 KAIROX_DUMP_ACTIVATION=1`.

### 4-1. 기준선 (`group_size=16`)

| 지표 | 값 | 의미 |
|---|---|---|
| `hit / resident` | 33.53% | GPU 슬롯 중 실제로 쓰인 비율 |
| `hit / activation` | 60.89% | 필요한 뉴런 중 GPU에 있던 비율 (39%는 CPU 경로) |
| `wasted / total` | 45.14% | 리밸런싱 로드 중 한 번도 안 쓰이고 축출된 비율 |

### 4-2. 무작위 배치 대비 이득 (lift)

`P(active)`는 아무 뉴런이나 골랐을 때 활성일 확률이므로, 정적·무작위 배치의 기대 적중률과
같다. DFR이 달성한 `P(active | resident)`와의 비가 리밸런싱의 순이득이다.

| | P(active) | P(active \| resident) | lift |
|---|---|---|---|
| 전체 | 23.61% | 33.53% | **1.420** |
| Layer 0 | 12.21% | 9.11% | **0.746** |
| Layer 1 | 37.63% | 46.15% | 1.226 |
| Layer 31 | 10.82% | 20.39% | 1.884 |

매 토큰 수천 번의 로드를 수행하고 얻는 이득이 무작위 배치 대비 **1.42배**에 그친다.

**레이어 0은 무작위 배치보다 나쁘다(0.746).** 원인은 구조적이다. `llama-graph.cpp:1400`에서
마지막 레이어의 리밸런싱 대상이 레이어 0으로 wrap되는데, 이때 쓰는 예측은 토큰 `t`의 것이고
실제 사용은 토큰 `t+1`이라 한 스텝 stale이다. 게다가 계측이 `active_t`로 결정한 마스크를 다시
`active_t`로 채점하므로 **0.746은 상한이고 실제는 더 나쁘다.**

이 값은 결정 단위를 좁히면 완화된다 — `group_size`를 64 → 16 → 4 → 1로 바꾸면 레이어 0의
lift가 0.680 → 0.746 → 0.991 → 1.005로 회복된다. 거친 그룹일수록 stale 예측의 피해가 크다.

### 4-3. 정적 배치 상한과의 비교

각 레이어에서 활성화 빈도 상위 `n_cached`개를 고정 배치했을 때의 적중률:

| Layer | DFR | 정적 배치 | 차이 |
|---|---|---|---|
| L0 | 30.80% | 77.51% | +46.7%p |
| L8 | 76.38% | 84.51% | +8.1%p |
| L15 | 59.16% | 77.16% | +18.0%p |
| L25 | 35.45% | 50.62% | +15.2%p |
| L30 | 36.94% | 53.32% | +16.4%p |

**32개 레이어 전부에서 정적 배치가 이긴다.** 그것도 PCIe 전송 0회로.

⚠️ 단 이는 **해당 실행의 활성화 빈도를 미리 아는 oracle**이므로 정적 배치의 *상한*이며
실측이 아니다. 공정한 비교가 되려면 프롬프트 A로 프로파일하고 프롬프트 B로 평가해야 한다.
그럼에도 격차가 8~47%p로 크다는 점은 DFR의 동적 정책이 정적 배치 대비 뚜렷한 이득을 내지
못하고 있음을 시사한다.


## 5. 문제 재정의 — 결정 단위와 전송 단위의 결합

### 5-1. 진단

`group_size=16`이라는 값 하나가 세 가지 역할을 동시에 수행한다:

1. DFR 점수 계산 단위 (`dfr_scores`, `group_mask`)
2. evict/load 결정 단위 (`load_group` / `evict_group` 비트벡터)
3. 물리적 DMA 전송 단위 (`ggml-cuda.cu`의 `group_nbytes`)

**1·2는 작을수록 정확하고, 3은 클수록 전송 효율이 높다.** 현재 설계는 이 상충하는
요구를 하나의 상수로 강제 결합해, 결정 단위를 전송 편의에 종속시키고 있다.

### 5-2. group_size 스윕 — 결정 단위를 실제로 바꿔보면

`ffn_reorder_perms`는 group_size와 무관한 순수 뉴런 재배치이므로, model-split GGUF의
`ffn_group_size` 스칼라만 교체하면(`regroup_model_split.py`) 같은 순열을 다르게 끊어
읽을 수 있다. 클러스터링 재실행이 필요 없다.

| gs | n_group | hit/act | wasted/total | 뉴런 로드 | PCIe(GB) | 전송 횟수 | decode |
|---|---|---|---|---|---|---|---|
| 1 | 11,008 | 60.99% | **19.73%** | 1,602,541 | 20.92 | 4,807,623 | 5.23 tok/s |
| 2 | 5,504 | 60.98% | 28.26% | 1,617,244 | 21.11 | 2,425,866 | 8.54 tok/s |
| 4 | 2,752 | 59.56% | 34.92% | 1,887,252 | 24.64 | 1,415,439 | 12.77 tok/s |
| 8 | 1,376 | 60.06% | 37.55% | 2,636,600 | 34.42 | 988,725 | 15.03 tok/s |
| **16** | 688 | 60.89% | 45.14% | 4,376,512 | 57.14 | 820,596 | 15.86 tok/s |
| 32 | 344 | 62.26% | 49.11% | 6,987,744 | 91.23 | 655,101 | 17.00 tok/s |
| 64 | 172 | 61.50% | **50.12%** | 7,283,264 | 95.09 | 341,403 | 19.35 tok/s |

세 가지가 드러난다.

**① 적중률은 granularity와 무관하다.** `hit/activation`이 64배 범위에 걸쳐
59.56~62.26%로 평평하다. 결정 단위를 잘게 해도 예측이 더 맞지는 않는다.

**② 그런데 낭비와 데이터 이동량은 급감한다.** `wasted/total`은 50.12% → 19.73%로
2.5배 줄고, 실제 옮기는 바이트는 95.09 GB → 20.92 GB로 **4.5배** 줄어든다.

**③ 그럼에도 처리량은 나빠진다.** 19.35 → 5.23 tok/s. 전송 횟수가 341K → 4.8M으로
14배 늘기 때문이다.

즉 **`group_size=16`은 정확도를 위한 선택이 아니다.** 정확도는 어차피 동일하고,
전송 횟수를 줄이려고 불필요한 뉴런까지 끌어오는 대가를 치르는 구조다.

### 5-3. decode 시간 분해 — 병목은 대역폭이 아니라 호출 횟수

바이트 수와 시간이 반대로 움직인다(gs=64는 95 GB를 26.4초, gs=1은 21 GB를 97.7초).
대역폭이 병목이면 정반대여야 한다. `time = 계산 + 전송횟수 × 오버헤드 + 바이트/대역폭`
으로 7개 지점을 최소제곱 적합하면:

```
time = 16,330 ms + 전송횟수 x 16.86 us + (바이트 항은 무시할 수준)     R^2 = 0.9984
```

| gs | 실측 | 계산 | 전송 오버헤드 | 대역폭 |
|---|---|---|---|---|
| 1 | 97,660 ms | 16,330 | **81,054** | 777 |
| 4 | 40,011 | 16,330 | **23,864** | 915 |
| 16 | 32,212 | 16,330 | **13,835** | 2,121 |
| 64 | 26,413 | 16,330 | **5,756** | 3,530 |

**느려진 원인의 사실상 전부가 호출 횟수다.** 대역폭 기여는 전체의 1~3%에 불과하다.

원인은 `ggml-cuda.cu:2651`의 `kairox_batch_reload`에 있다. 이름과 달리 그룹마다 별도의
`cudaMemcpyAsync`를 순차 호출한다. `reload_window_size = 4`이므로 전송 4번마다
`cudaStreamSynchronize`도 들어간다.

⚠️ **동기화 주기는 원인이 아니었다(5-7절에서 반증).** `KAIROX_RELOAD_WINDOW`로 주기를
4 → 16 → 64로 올려도 end-to-end 처리량은 움직이지 않는다(13.78 → 14.37 → 14.10 tok/s,
런간 변동 범위 안). reload는 `SingleThreadExecutor`에서 계산과 겹쳐 돌기 때문에, 그 스레드
내부의 동기화 빈도는 계산이 여유를 남기는 한 임계 경로에 나타나지 않는다. 지배적인 것은
**호출 그 자체의 개수**다.

### 5-4. PCIe 마이크로벤치마크 — 위 해석의 독립 검증

`bench_pcie.cu`로 전송 크기와 동기화 주기를 직접 재면 (RTX 3070):

**A. 68 KB 전송(현재 gs=16 조각 크기)에서 동기화 주기별**

| sync 주기 | 호출당 | 실효 대역폭 |
|---|---|---|
| 매번 | 40.12 us | 1.74 GB/s |
| **4회마다 (현재 KAIROX)** | **14.64 us** | **4.76 GB/s** |
| 16회마다 | 7.70 us | 9.04 GB/s |
| 끝에 한 번 | 5.92 us | 11.76 GB/s |

측정된 14.64 us는 end-to-end 적합값 16.86 us와 잘 맞는다(차이는 executor 스레드
핸드오프). 동기화 주기만 바꿔도 **전송 경로 자체는** 2.5배 빨라진다.

⚠️ 다만 이것은 다른 일이 없는 상태에서 잰 전송 스레드의 처리량이다. 실제 시스템에서
reload는 계산과 겹쳐 돌므로 이 이득은 end-to-end 로 넘어오지 않았다(5-7절). 이 표는
"호출당 고정비용이 실재한다"는 근거이지 "동기화를 줄이면 빨라진다"는 근거가 아니다.

**B. 대역폭 포화 지점 (끝에 한 번만 동기화)**

| 전송 크기 | 대역폭 |
|---|---|
| 4.2 KB | 0.91 GB/s |
| 68 KB | 11.76 GB/s |
| 272 KB | 22.12 GB/s |
| **1 MB** | **28.19 GB/s** |
| 4 MB | 27.16 GB/s |

**약 1 MB에서 28 GB/s로 포화**한다. 뉴런 1행이 4352 B이므로 **한 번에 240행 이상**을
모아야 피크에 근접한다. 이것이 gather 배치 크기의 하한이다.

**C. 흩어진 행: 개별 전송 vs gather 후 일괄 전송**

| 행 수 | 개별 전송 | gather + 1회 | 배속 |
|---|---|---|---|
| 1,024 | 12.32 ms | 0.74 ms | **16.6x** |
| 4,096 | 41.65 ms | 2.64 ms | **15.8x** |
| 16,384 | 164.91 ms | 11.40 ms | **14.5x** |

CPU 측 gather(단일 스레드 memcpy)가 비용의 대부분이지만(16,384행에서 11.4 ms 중 8.9 ms),
그래도 개별 전송보다 14~20배 빠르다. 멀티스레드로 더 줄일 여지도 있다.

### 5-5. 두 개의 독립적인 문제 — 정책과 전송

`group_size=1`은 **뉴런 단위 결정**을 의미한다. DFR이 EMA 점수 상위 N개 뉴런을 정확히
고른다는 뜻이고, 그룹 경계로 인한 손실이 원리적으로 존재하지 않는다. 그런데도 이때
`hit/activation`은 **60.99%**에 머문다.

즉 **granularity는 애초에 정확도의 병목이 아니었다.** 결정 단위를 아무리 잘게 해도
EMA 정책 자체가 약 61%를 천장으로 갖는다. 반면 같은 캐시 용량에서 활성화 빈도 상위
N개를 고정 배치하면(4-3절) 70~85%에 도달한다.

따라서 관측된 문제는 서로 독립적인 두 가지로 분리된다.

| | 증상 | 원인 | 해법 | 상태 |
|---|---|---|---|---|
| **전송 구조** | 같은 적중률에 4.5배 데이터 이동, 호출 오버헤드가 시간의 대부분 | `group_size`가 결정 단위와 DMA 단위를 겸함 | gather/scatter로 분리 | **구현·측정 완료 (5-6, 5-7절)** |
| **예측 정책** | 뉴런 단위로 완벽히 골라도 적중률 61% 천장 | EMA 기반 DFR 점수 | 다른 정책 필요 | **미탐색** |

두 문제는 곱해지는 관계가 아니라 더해지는 관계다. gather를 구현하면 **전송 비용**이
줄지만 적중률은 61%에 머물고, 더 나은 정책을 찾으면 **적중률**이 오르지만 거친 그룹에서는
여전히 불필요한 데이터를 옮긴다. 둘 다 해결해야 정적 배치를 확실히 앞선다.
실제로 gather를 구현해 측정한 결과가 이 분리를 뒷받침한다 — 전송 경로만 바꿨더니
정책 지표(hit/act, wasted)는 런간 변동 범위 안에서 그대로였다(5-7절).

레이어 0의 lift가 결정 단위에 따라 0.680 → 1.005로 회복되는 것(4-2절)도 이 분리로 설명된다.
wrap-around로 인한 한 스텝 stale이라는 *정책* 문제가, 거친 *전송* 단위와 만나 증폭되는
구조다. 전송 단위를 좁히면 증폭분은 사라지지만 stale 자체는 남아 lift가 1.0 부근에 그친다.

이 문서의 나머지는 전송 구조 문제를 다룬다. 예측 정책 문제는 별도 과제다.

### 5-6. 해법 — gather 버퍼로 결정 단위와 전송 단위 분리

결정은 뉴런 단위로 내리되, 실행 시점에 선택된 뉴런들이 원본에서 흩어져 있더라도
pinned staging buffer에 모아 하나의 큰 연속 전송으로 보낸다.

```
[결정]   뉴런별 DFR 점수 -> 이번 스텝 로드할 K개 뉴런 (원본에서 흩어져 있음)
            |
[gather] CPU: 선택된 K개 행을 pinned staging buffer 에 연속 복사
            |
[전송]   cudaMemcpyAsync 1회, K x 4352 B (K >= 240 이면 대역폭 포화)
            |
[scatter] GPU: 커널로 staging buffer -> 각 캐시 슬롯에 분산 기록
            |
[메타]   neuron_idx / neuron_mask 갱신 (이미 뉴런 단위로 존재)
```

인프라의 절반은 이미 있다. `neuron_mask`, `neuron_idx`, 계측 카운터가 전부 뉴런
단위이고 `SingleThreadExecutor`가 비동기 실행 골격을 제공한다. 새로 필요한 것은
(a) DFR 스코어와 그룹 마스크의 뉴런 단위 일반화, (b) gather/scatter 커널,
(c) staging buffer 관리다.

**구현.** `ggml/src/ggml-cuda/kairox-gather.cu`. `KAIROX_GATHER=1`로 켜고, 끄면 기존
경로를 그대로 탄다.

목적지 슬롯(`cache_base + slot_idx * nbytes`)도 불연속이므로 host gather만으로는 부족하다.
staging 버퍼는 연속이지만 캐시 슬롯은 흩어져 있어 디바이스 쪽 scatter 커널이 필요하다.
따라서 한 청크당 CUDA 호출은 3회다 — 가중치 H2D 1회, 슬롯 인덱스 H2D 1회, scatter 커널 1회.
실행기가 전역에 하나이고 태스크가 직렬화되므로 staging 버퍼도 전역 하나를 재사용한다.

노브: `KAIROX_GATHER_BUDGET_MIB`(기본 64, 초과 시 청크 분할),
`KAIROX_GATHER_VERIFY`(되읽기 바이트 검증), `KAIROX_RELOAD_WINDOW`(기존 경로의 동기화 주기).

**정확성 검증.** KAIROX는 reload가 계산과 비동기로 경쟁하므로 **같은 seed로도 생성 텍스트가
런마다 다르다**(baseline을 두 번 돌려 확인: 계측 CSV가 서로 불일치). 따라서 출력 비교로는
전송 경로의 정확성을 검증할 수 없다. 대신 scatter 직후 GPU 캐시 슬롯을 되읽어 원본 가중치와
`memcmp`하는 검증기를 넣었다 — **80,030개 그룹 전송, 불일치 0.**

정책 지표도 baseline의 런간 변동 범위 안에 들어온다(전송만 바꿨으므로 당연해야 한다):

| 런 | hit/res | hit/act | wasted/total |
|---|---|---|---|
| baseline #1 | 37.75% | 56.23% | 46.90% |
| baseline #2 | 38.45% | 55.42% | 45.33% |
| baseline #3 | 37.49% | 56.07% | 46.06% |
| **gather** | **37.42%** | **56.16%** | **46.68%** |

### 5-7. gather 측정 결과 — 전송 단위가 결정 단위에서 분리된다

계측을 끄고(`DUMP=0`, 계측은 레이어마다 D2H + 동기화를 추가한다) 511 decode 토큰,
`--ignore-eos`, RTX 3070 / vb=6 GiB. `bash gather_ab.sh`.

**A. decode 처리량 (tok/s, 평균 ± 표준편차)**

| group_size | baseline | gather | 배속 |
|---|---|---|---|
| 1  | 6.31 ± 0.04 (n=2) | 14.26 ± 1.46 (n=2) | **2.26x** |
| 4  | 11.36 ± 1.23 (n=2) | **17.23 ± 0.23** (n=4) | **1.52x** |
| 16 | 13.78 ± 1.79 (n=4) | 17.68 ± 1.06 (n=4) | **1.28x** |
| 64 | 17.74 ± 1.13 (n=4) | 17.28 ± 0.52 (n=4) | 0.97x |

**gather의 이득은 결정 단위가 잘수록 커진다.** 호출 횟수가 병목이라는 진단과 정확히
일치한다 — gs=64는 이미 호출이 적어 이득이 없고(0.97x, 변동 범위 안), gs=1에서 2.26배다.

**핵심은 배속이 아니라 수렴이다.** gather를 켜면 gs=4/16/64가 전부 **17.2~17.7 tok/s**로
모인다. PCIe 전송량이 24.6 GB에서 95.1 GB로 3.9배 차이나는데도 처리량이 같다.
**전송 단위가 결정 단위로부터 분리된 것이다.**

**B. 동기화 주기 (gs=16, gather 없이)**

| `reload_window_size` | tok/s |
|---|---|
| 4 (저자 기본값) | 13.78 (n=4) |
| 16 | 14.37 (n=1) |
| 64 | 14.10 (n=1) |

**변화 없음.** 5-4절의 마이크로벤치마크는 전송 스레드를 고립시켜 쟀기 때문에 2.5배가
나왔지만, 실제로는 그 스레드가 계산과 겹쳐 돌아 여유가 있다. 동기화 빈도가 아니라
호출 개수가 문제였다.

**C. 실무적 의미**

같은 처리량에서 캐시 거동이 훨씬 낫다:

| 구성 | tok/s | PCIe | wasted/total | L0 lift |
|---|---|---|---|---|
| baseline 최선 (gs=64) | 17.74 | 95.1 GB | 50.12% | 0.680 |
| **gather + gs=4** | **17.23** | **24.6 GB** | **34.92%** | **0.991** |

처리량은 같고(차이가 표준편차 안), PCIe 트래픽 **3.9배 감소**, 낭비 적재 **1.4배 감소**,
레이어 0의 lift가 0.680에서 0.991로 회복된다. 즉 **gather는 천장을 올리는 것이 아니라
fine-grained 결정을 감당 가능하게 만든다.** 이것이 "결정 단위와 전송 단위의 분리"가
실제로 뜻하는 바다.

**D. 예측이 빗나간 부분 — 정직한 기록**

이 절 이전 판에는 적합 모델(`time = 16,330 ms + 전송횟수 x 16.86 us`)로 gather + gs=1이
**26~30 tok/s**에 이를 것이라 적혀 있었다. 실측은 **14.3 tok/s**다. 두 군데가 틀렸다.

1. 그 모델은 전송 시간이 **전부 임계 경로에 있다**고 가정했다. 실제로는 reload가
   `SingleThreadExecutor`에서 계산과 겹쳐 돌아 상당 부분이 숨는다. gs=16에서 전송
   오버헤드는 토큰당 약 27 ms로 계산되지만 gather로 실제 회수된 것은 약 10 ms뿐이다 —
   나머지 17 ms는 이미 계산 뒤에 숨어 있었다.
2. gs=1에는 전송과 무관한 비용이 따로 붙는다. `group_identity`가 462 MiB를 차지해 캐시
   용량을 잠식하고, DFR top-k가 11,008개 그룹에 대해 CUB 경로로 폴백하며, 4,352 B짜리
   조각 11,008개를 host에서 모으는 memcpy가 바이트당 비효율적이다. 그래서 gather를
   켜도 gs=1(14.26)이 gs=4(17.23)보다 느리다.

남은 천장 약 17.5 tok/s는 전송이 아니다 — PCIe 트래픽이 3.9배 달라져도 같은 값이 나온다.
계산(sparse matmul, predictor) 또는 정책 쪽이며, 다음 조사 대상이다.

### 5-8. 스윕에 필요했던 코드 수정

결정 단위를 16보다 잘게 내리려면 두 겹의 1024 제약을 풀어야 했다.

1. `llama-kairox.cpp:230` `GGML_ASSERT(n_group <= 1024)` — 저자의 성능 가이드라인이다
   ("Recommended"). `ggml_argsort_top_k`는 `ncols > 1024`에서 CUB device-wide sort로
   폴백하므로(`top-k.cu:81`, `GGML_CUDA_USE_CUB`는 CUDART >= 11.7에서 정의) 기술적
   한계가 아니다. 경고로 완화했다.
2. `dfr-fusion.cu:219` `GGML_ASSERT(n_groups <= 1024)` — 이쪽이 실제 제약이었다.
   `kairox_dfr_mask_f32_kernel`이 고정 크기 공유 메모리 비트마스크
   (`curr_mask_bits[1024/32]`)를 쓰기 때문이다. 그룹당 1비트뿐이므로 상한을 16384
   그룹으로 넓혔다(공유 메모리 2 KiB, 블록당 48 KiB 한도에 한참 못 미침).

`group_identity`(`n_group x n_group` F32)는 gs=1에서 462 MiB까지 커지지만 8 GiB
카드에서 동작을 확인했다. 다만 DFR fusion이 활성일 때 이 텐서의 데이터는 읽히지
않으므로(`ggml_cuda_op_dfr_mask`는 `topk_idx`만 받는다) 제거 여지가 있다.

### 5-9. 통제의 한계

`--ignore-eos`로 생성 길이는 511 토큰으로 고정했으나 **텍스트 자체의 발산은 남는다.**
group_size가 바뀌면 GPU 상주 집합이 바뀌고, GPU 경로와 CPU 경로의 부동소수점 누적
순서가 달라 미세한 수치 차이가 샘플링을 거쳐 증폭되기 때문이다. 실제로 `P(active)`가
런마다 22.2~24.6%로 변동한다.

핵심 지표(`hit/act` 평탄, `wasted` 단조 감소, 전송 횟수 지배)는 이 변동폭보다 훨씬 큰
차이를 보이므로 결론은 견고하다. 다만 엄밀한 통제가 필요하면 활성화 trace를 한 번
기록하고 정책을 오프라인에서 재생하는 방식(trace 기반 시뮬레이션)이 필요하다.


## 6. 정책 층 — 논문 대조와 lambda 스윕

5장은 전송 구조를 다뤘고 결론은 "전송은 병목이 아니다"였다. 그러면 무엇이 병목인가.
상주하지 못한 active 뉴런은 CPU 가 계산하므로(`llama-graph.cpp:1553`), **적중률이 곧
CPU 작업량**이다. 이 장은 그 적중률을 정하는 정책을 논문과 대조한다.

### 6-1. 논문 Algorithm 1 과 구현의 차이

| 논문 | 구현 | |
|---|---|---|
| `Feedback ← GetSystemBottleneck()` | 없음 — `IO_BOUND`/`CPU_BOUND` 판정 코드가 저장소에 존재하지 않는다 | ❌ |
| `λ ← min(λ(1+α), λmax)` (IO-bound) | 없음 | ❌ |
| `λ ← max(λ(1−α), λmin)` (CPU-bound) | 없음 | ❌ |
| `λmin`, `λmax` | 없음 — 심볼 자체가 없다 | ❌ |
| `S ← λS + (1−λ)A` | `llama-graph.cpp:1315` | ✅ |
| `τload = (1−λ) + ε` | `llama-graph.cpp:1327`, 그룹 단위만 (논문은 뉴런 단위 우선), ε=1e-6 | ⚠️ |
| `TopK(Ccand, K)` | 걸러진 그룹에 0을 곱한 뒤 전체를 argsort — 통과 그룹이 K 미만이면 점수 0 짜리가 임의로 로드된다 | ⚠️ |

**λ 는 런타임에 절대 바뀌지 않는다.** `dfr_ema_coeffs` 에 쓰는 곳은 생성자 안
`llama-kairox.cpp:439` 한 줄뿐이고, 32 개 레이어가 같은 상수를 공유한다. 논문 Figure 11 은
레이어마다 λ 가 다르게 수렴하는 그래프인데(L16/L31 은 0.85, L0 은 더 낮게) 전역 상수로는
원리적으로 재현할 수 없다.

`KAIROX_DFR_LAMBDA_ADAPT_RATE` 는 이름과 달리 **λ 를 건드리지 않는다.**

```cpp
// ggml-kairox.hpp:251 — 조절 대상은 K(스텝당 최대 스왑 수)다
int nxt = cur * (1 + (to_move.empty() ? +rate : -rate));
dfr_clamp_k->store(clamp(nxt, 1, cap));
```

| | 논문 | 구현 |
|---|---|---|
| 신호 | 파이프라인 stall 로 IO/CPU bound 판정 | `to_move.empty()` — 앵커에 밀린 태스크가 있었나 |
| 제어 변수 | λ (decay factor) | K (reload 예산) |
| 파급 | TAM 관성 **+** τload 임계 동시 이동 | reload 개수 상한만 |

즉 negative feedback loop 은 있으나 다른 변수를 다른 신호로 제어하며, **스코어링 정책
자체는 고정**이다. 논문 어블레이션(§9.4)은 `+NB`(고정 λ=0.5) 3.32x/2.88x 에서
`+ANB`(feedback λ) 3.70x/3.46x 로 올라간다고 보고한다. 구현된 것은 `+NB` 까지다.

### 6-2. lambda 스윕 — 기본값이 잘못 조율돼 있다

gather 켜고 `-t 4`, 511 decode 토큰, 계측 ON(지표 수집용).

| λ | τ=(1−λ)+ε | tok/s | hit/act | wasted/total | reload | PCIe |
|---|---|---|---|---|---|---|
| 0.50 (논문 초기값) | 0.500 | 20.00 | 58.46% | 43.31% | 7,590,688 | 99.10 GB |
| **0.67 (아티팩트 기본값)** | 0.330 | 20.62 | 64.51% | 48.23% | 8,041,088 | 104.98 GB |
| 0.85 (논문 ANB 수렴값) | 0.150 | 22.56 | 66.23% | 43.49% | 3,820,800 | 49.88 GB |
| 0.95 | 0.050 | 25.05 | 67.11% | 34.17% | 1,233,440 | 16.10 GB |
| 0.98 | 0.020 | 24.46 | 66.35% | 29.63% | 627,216 | 8.19 GB |
| **0.99** | 0.010 | **27.39** | **69.08%** | 28.37% | 422,688 | 5.52 GB |
| 0.995 | 0.005 | 25.68 | 66.07% | 24.15% | 375,296 | 4.90 GB |
| 0.999 | 0.001 | 25.11 | 64.51% | 21.68% | 327,072 | 4.27 GB |

계측을 끄고 3회씩 반복한 확인:

| λ | 런 | 평균 |
|---|---|---|
| 0.67 | 19.89 / 21.40 / 20.90 | 20.73 ± 0.77 |
| **0.99** | 26.10 / 25.05 / 26.45 | **25.87 ± 0.73** |

**+24.8%, 구간이 겹치지 않는다.**

세 가지가 동시에 나온다.

1. 아티팩트의 고정값 0.67 은 **논문 자신의 수렴값 0.85 보다도 나쁘다**(20.62 vs 22.56).
   ANB 미구현은 단순 누락이 아니라 측정 가능한 손해다.
2. 논문 초기값 0.5 가 스윕 최악이다. ANB 가 있으면 거기서 출발해 올라가겠지만,
   ANB 가 없는 아티팩트에서 0.5 로 돌리면 최악에 고정된다.
3. 최적은 0.99 로 논문 수렴값보다 훨씬 높다. Figure 11 의 λ 는 0.6~0.85 구간에서
   진동하므로 ANB 가 있어도 도달하지 못하는 영역일 수 있다.

### 6-3. 내부 최적점이 있다 — 동적 리밸런싱 자체는 유효하다

λ→1 이면 점수가 얼어붙어 초기 배치(빈도 기반 `ffn_normalized_pattern`)가 유지된다.
사실상 정적 배치다. 그런데 λ=0.999 는 λ=0.99 보다 나쁘다(25.11 vs 27.39, 적중률
64.51% vs 69.08%). **내부 최적점이 실재하므로 "리밸런싱을 하지 마라"가 결론은 아니다.**

다만 최적점의 reload 양은 아티팩트 기본값의 **1/19** 이다. 그리고 λ=0.999 의 적중률
64.51% 는 λ=0.67 과 정확히 같다 — **105 GB 를 옮겨 얻는 적중률이 4.3 GB 만 쓸 때와 같다.**

### 6-4. Challenge 3 의 전제가 이 구간에서 성립하지 않는다

논문 §7 은 I/O 지연과 CPU 부하의 **역상관**을 전제한다.

| λ | I/O (PCIe) | CPU 부하 (= 1 − hit/act) |
|---|---|---|
| 0.67 | 105.0 GB | 35.5% |
| 0.99 | 5.5 GB | **30.9%** |

**둘 다 좋아진다.** 트레이드오프가 아니라 파레토 개선이며, 역상관은 λ > 0.99 부터만
나타난다(0.999 에서 I/O 4.3 GB, CPU 35.5%). 논문이 해결하려는 "coupled bottleneck"은
저자가 고른 동작점 근처에서만 존재하는 문제로 보인다.

또한 논문의 제어 방향이 실측과 반대다.

> CPU-Bound → `λ ← max(λ(1−α), λmin)` — *"increases balancing intensity, reducing CPU workload"*

λ 를 낮추면 reload 는 폭증하지만(1.2M → 8.0M) **적중률이 떨어진다**(67.11% → 64.51%).
I/O 만 늘고 CPU 부하는 오히려 는다. λ 가 관성과 임계값(τ=(1−λ)+ε)을 동시에, 반대
방향으로 제어하기 때문이다. 논문은 앞의 효과만 논증하고 뒤의 것은 언급하지 않는다.

⚠️ 이 결과는 λ 최적값이 워크로드에 의존한다는 **논문의 논거를 오히려 강화한다.** 우리가
찾은 0.99 도 프롬프트 하나·모델 하나·하드웨어 하나에서 나온 값이다. 비판의 대상은
"적응이 필요하다"는 주장이 아니라 **아티팩트가 그 적응을 구현하지 않은 채 나쁜 상수를
고정했다**는 사실이다.

### 6-5. 프롬프트 길이 — 전송 병목을 만들려는 마지막 시도

"granularity 는 전송이 포화하는 영역에서만 의미를 갖는다"는 가설을 시험했다.

먼저 **512 토큰이 넘는 프롬프트는 크래시했다**(`llama-graph.cpp:1420`, 커밋 `a82951b47`
에서 수정). 마지막이 아닌 ubatch 는 `n_outputs == 0` 이라 마지막 레이어의 `sparse_idx` 가
`[n_ff, 0]` 이 되고 `reshape_2d` 단언이 깨진다. 같은 351 토큰 프롬프트가 ubatch=512(1개)면
통과, ubatch=128(3개)면 abort 한다 — 길이가 아니라 분할 여부가 트리거다.

수정 후 측정하면 가설과 정반대다.

| | forward pass | 총 reload | pass 당 | pass 소요 | PCIe 전송률 |
|---|---|---|---|---|---|
| prefill 2458 토큰 | 5 (ubatch) | 27,520 | 5,504 | 5.5 s | **0.013 GB/s** |
| decode 128 토큰 | 128 | 1,111,136 | 8,681 | 0.057 s | 2.0 GB/s |

**rebalancing 은 토큰당이 아니라 forward pass 당 한 번**이므로, prefill 은 512 토큰을
pass 하나로 묶어 rebalancing 기회를 512 분의 1 로 줄인다. batch 를 키우면 전송률이
**내려간다.** 짧은 프롬프트(1 ubatch) + 1 decode 토큰은 reload 가 0 회였다.

prefill 에서는 gather 도 손해다(89.05 → 83.76 tok/s). 호출이 적어 이득이 없고 host
memcpy 비용만 붙는다.

### 6-6. gather 손익분기가 독립 상수로 예측된다

5-4 절의 마이크로벤치마크 상수 두 개만으로 5-7 절의 실측 교차점이 나온다.

```
호출당 오버헤드 16.86 us  >  그룹 바이트 / gather 대역폭 8.02 GB/s
→ 손익분기 그룹 크기 = 16.86e-6 x 8.02e9 = 약 135 KB
```

| gs | 그룹 바이트 | 예측 | 실측 |
|---|---|---|---|
| 4 | 17 KB | gather 유리 | 11.36 → 17.23 ✅ |
| 16 | 68 KB | gather 유리 | 13.78 → 17.68 ✅ |
| 32 | 136 KB | 손익분기 | — |
| 64 | 279 KB | gather 불리 | 17.74 → 17.28 ✅ |

**gather 는 CUDA 호출 오버헤드를 host memcpy 대역폭과 맞바꾼다.** 5-7 절의 적합 모델과
달리 이 예측에는 "전송이 임계 경로에 있다"는 가정이 필요 없다. prefill 이 예외가 아니라
확장인 것도 같은 규칙으로 설명된다 — prefill 은 CPU 가 dense FFN 으로 포화 상태라
실질 gather 대역폭이 낮아 손익분기가 왼쪽으로 이동한다.

### 6-7. 누적 결과

| 단계 | 변경 | tok/s |
|---|---|---|
| 저자 기본값 | baseline 전송, t=7, λ=0.67 | 13.78 ± 1.79 (n=4) |
| + 설정 | t=4 (오버서브스크립션 제거) | 17.22 ± 0.53 (n=3) |
| + 전송 | gather | 20.11 ± 0.70 (n=3) |
| + 정책 | λ=0.99 | **25.87 ± 0.73** (n=3) |

**1.88배.** 새 알고리즘은 하나도 없다 — 오버서브스크립션 제거, 전송 묶기,
하이퍼파라미터 하나. TAM 자체는 유효하다(λ=0.99 가 정적 배치보다 낫다).
부정되는 것은 동작점 선택과 그것을 전제로 한 Challenge 3 서사다.

⚠️ 프롬프트 하나, 모델 하나(prosparse-llama-2-7b Q8_0), 하드웨어 하나(RTX 3070,
PCIe 4.0 x16, 8 코어). 특히 `t=7 → 4` 는 8 코어 기기 고유일 수 있다 — 논문 플랫폼은
3080Ti(12 스레드), 4090(16 스레드)로 오버서브스크립션 비율이 낮다.

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
