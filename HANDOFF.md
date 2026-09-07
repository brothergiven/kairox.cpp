# 작업 인수인계 — `exp/group-granularity`

KAIROX의 Wasted Rebalancing 계측, group_size 스윕, gather/scatter 구현·측정까지 끝난 시점의 상태 정리.
다른 환경에서 이어받을 때 이 문서부터 읽으면 된다. 상세 결과는 [README.md](README.md) 4·5절.

---

## 1. 한 줄 요약

KAIROX의 `group_size=16`은 정확도를 위한 선택이 아니다. 적중률은 결정 단위와 무관하게
약 61%로 일정하고, 거친 그룹은 **같은 적중률을 얻기 위해 4.5배 많은 데이터를 옮긴다.**
그런데 decode 시간의 지배 요인은 PCIe 대역폭이 아니라 **전송 호출 횟수**였다.

gather/scatter 를 구현해 확인했다: **gather 는 처리량 천장을 올리지 않는다. fine-grained
결정을 감당 가능하게 만든다.** gather 를 켜면 gs=4/16/64 가 전부 17.2~17.7 tok/s 로
수렴한다 — PCIe 전송량이 3.9배 차이나는데도. gs=4 + gather 는 baseline 최선(gs=64)과 같은
처리량을 내면서 PCIe 3.9배, 낭비 적재 1.4배가 적다.

---

## 2. 커밋

```
1dcad98ed  docs: gather 실측 결과 반영, 동기화 주기 가설 반증 기록
318cb55d8  feat: gather/scatter reload 경로 구현 및 A/B 하네스 추가
254ddb287  docs: 통제 런 기준으로 수치 통일, gather 비용 반영, 정책/전송 문제 분리
01c75be5d  feat: PCIe 마이크로벤치마크 추가 및 전체 결과로 README 재작성
221d744f9  feat: DFR mask 커널의 그룹 상한을 1024 -> 16384 로 확대
b0ea47b94  fix: dump_activation.sh 의 set -o pipefall 오타 수정
8fb294e44  feat: n_group 단언 완화 및 group_size 4~64 스윕 준비
```

`main`(계측 코드까지 포함) 위에 얹혀 있다. C++ 변경은 두 파일 22줄뿐이고 나머지는
스크립트와 문서다.

| 파일 | 성격 |
|---|---|
| `src/llama-kairox.cpp` | `GGML_ASSERT(n_group <= 1024)` → 경고로 완화 |
| `ggml/src/ggml-cuda/dfr-fusion.cu` | DFR mask 커널 그룹 상한 1024 → 16384 |
| `regroup_model_split.py` | model-split GGUF의 `ffn_group_size`만 교체 (신규) |
| `group_sweep.sh` | group_size 스윕 러너 (신규) |
| `bench_pcie.cu` | PCIe 전송 비용 마이크로벤치마크 (신규) |
| `dump_activation.sh` | `MODEL_SPLIT` / `BIN` / `IGNORE_EOS` 오버라이드 추가 |
| `README.md` | 1~5절 전면 갱신 |

---

## 3. 환경 재구축

### 3-1. 필요한 것

- CUDA 11.7 이상 (CUB 경로 필요). 개발 환경은 12.0
- NVIDIA GPU. 측정은 RTX 3070 8 GiB 에서 수행
- 호스트 RAM 12 GiB 이상 여유 — 모델이 pinned 4.5 GiB 를 잡는다 (스왑 불가)
- `libssl-dev`

### 3-2. 빌드

```bash
git clone <fork> && cd kairox.cpp
git checkout exp/group-granularity
bash compile_kairox.sh rel        # -> build_rel/bin/llama-completion
```

`compile_kairox.sh`는 `-DCMAKE_CUDA_ARCHITECTURES=native`를 쓴다. **다른 GPU로 옮기면
반드시 `rm -rf build_rel` 후 재빌드할 것.**

### 3-3. 모델

```bash
hf download Anhelor/SPIF-GGUF --repo-type model \
    --include "prosparse-llama-2-7b.gguf" \
    --include "prosparse-llama-2-7b-sparkinfer-model-split-688.gguf" \
    --local-dir ~/SPIF-GGUF

# Q8_0 재양자화 — 빌드 후에 실행 (llama-quantize 가 빌드 산출물)
./build_rel/bin/llama-quantize --tensor-type ffn_pred=f16 \
    ~/SPIF-GGUF/prosparse-llama-2-7b.gguf \
    ~/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf Q8_0
```

`--tensor-type ffn_pred=f16`은 필수다. 예측기를 양자화하면 `activation_count` 자체가
오염되어 계측이 무의미해진다.

### 3-4. group_size 변형 생성

```bash
for gs in 1 2 4 8 32 64 128; do
  python3 regroup_model_split.py ~/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf $gs
done
```

`ffn_reorder_perms`는 group_size와 무관한 순수 뉴런 재배치이므로 클러스터링 재실행이
필요 없다. 스칼라 하나만 바뀐다.

### 3-5. 실행

```bash
bash dump_activation.sh                                   # 단일 (gs=16 기본)
SIZES="1 2 4 8 16 32 64" bash group_sweep.sh              # 스윕
nvcc -O3 -std=c++17 -o bench_pcie bench_pcie.cu && ./bench_pcie
```

---

## 4. 결과 (RTX 3070, vb=6, 511 decode 토큰, `--ignore-eos`)

| gs | n_group | hit/act | wasted/total | PCIe(GB) | 전송 횟수 | decode |
|---|---|---|---|---|---|---|
| 1 | 11,008 | 60.99% | **19.73%** | 20.92 | 4,807,623 | 5.23 tok/s |
| 2 | 5,504 | 60.98% | 28.26% | 21.11 | 2,425,866 | 8.54 |
| 4 | 2,752 | 59.56% | 34.92% | 24.64 | 1,415,439 | 12.77 |
| 8 | 1,376 | 60.06% | 37.55% | 34.42 | 988,725 | 15.03 |
| **16** | 688 | 60.89% | 45.14% | 57.14 | 820,596 | 15.86 |
| 32 | 344 | 62.26% | 49.11% | 91.23 | 655,101 | 17.00 |
| 64 | 172 | 61.50% | **50.12%** | 95.09 | 341,403 | 19.35 |

CSV 원본은 `dumps/gs*.csv` (커밋하지 않음, 각 7~8 MB).

### 핵심 발견 셋

1. **적중률은 결정 단위와 무관하다** — 64배 범위에서 59.6~62.3%
2. **낭비와 전송량은 급감한다** — 50.12% → 19.73%, 95 GB → 21 GB
3. **그럼에도 처리량은 나빠진다** — 전송 횟수가 14배 늘기 때문

### 병목 규명

바이트와 시간이 **반대 방향**으로 움직인다(95 GB/26초 vs 21 GB/98초). 대역폭 가설로는
설명 불가능하다. 7점 최소제곱 적합:

```
time = 16,330 ms + 전송횟수 x 16.86 us + (바이트 항 기여 1~3%)     R^2 = 0.9984
```

원인은 `ggml-cuda.cu:2651` `kairox_batch_reload`가 그룹마다 별도 `cudaMemcpyAsync`를
호출하고 `reload_window_size = 4`마다 `cudaStreamSynchronize`를 거는 구조다.

마이크로벤치마크가 독립 검증: 68 KB 전송에서 4회마다 동기화 시 14.64 us,
끝에 한 번만 하면 5.92 us — **2.5배 차이.** 대역폭은 약 1 MB에서 28 GB/s 포화하므로
gather 배치는 240행 이상이어야 한다.

---

## 5. 증거의 층위 — 무엇이 실측이고 무엇이 추정인가

이 구분이 논문 서술에서 중요하다.

| 층위 | 항목 | 신뢰도 |
|---|---|---|
| **실측** | 7개 런의 decode 시간, 적중률, 낭비율 | 벽시계·카운터 직접 측정 |
| **유도** | PCIe 바이트, 전송 횟수 (`total_loads` × 상수) | 측정값 × 형상 상수. 바이트를 직접 계측한 것은 아님 |
| **실측(합성)** | 마이크로벤치마크의 14.64 us, 28 GB/s, gather 14~20배 | 진짜 측정이나 격리 환경. 추론 중 경합 미반영 |
| **모델** | `16,330 ms + 횟수 × 16.86 us` 분해 | 실측 7점에 대한 적합. 세 항 분해는 가정한 모델 형태 |
| ~~외삽~~ | ~~gather 적용 시 26~30 tok/s~~ | **반증됨. 실측 14.3 tok/s** — 아래 참조 |
| **실측** | gather A/B 4개 group_size x 2~4 반복 | `gather_ab.sh`, 계측 OFF |
| **실측** | gather 경로의 바이트 정확성 | 되읽기 memcmp 80,030건 불일치 0 |

**가장 강한 근거는 모델이 아니라 "바이트와 시간이 반대로 움직인다"는 관측이다.**
이건 어떤 모델 가정에도 의존하지 않는다.

**적합 모델은 실측으로 반증됐다.** `16,330 ms + 횟수 x 16.86 us` 는 전송 시간이 전부
임계 경로에 있다고 가정했는데, reload 는 `SingleThreadExecutor` 에서 계산과 겹쳐 돌아
상당 부분이 숨어 있었다. gs=16 에서 모델이 계산한 전송 오버헤드는 토큰당 약 27 ms 지만
gather 로 실제 회수된 것은 약 10 ms 다. 이 모델은 "호출 횟수가 지배한다"는 방향성의
근거로만 쓰고, 절대값 예측에는 쓰지 말 것.

---

## 6. 다음 단계

### 6-1. 완료 — `reload_window_size` 실측 (가설 반증)

`KAIROX_RELOAD_WINDOW` 로 노출했고 4 / 16 / 64 를 측정했다.
결과 13.78 / 14.37 / 14.10 tok/s — **변화 없음.** 예측했던 21.3 tok/s 는 틀렸다.

마이크로벤치마크의 2.5배는 전송 스레드를 고립시켜 잰 값이다. 실제로는 그 스레드가
계산과 겹쳐 돌아 여유가 있으므로 내부 동기화 빈도가 임계 경로에 나타나지 않는다.
지배 요인은 동기화 주기가 아니라 **호출 개수 자체**다.

### 6-2. 완료 — gather/scatter 구현 (`ggml/src/ggml-cuda/kairox-gather.cu`)

`KAIROX_GATHER=1`. 실측 결과는 README 5-7절.

| gs | baseline | gather | 배속 |
|---|---|---|---|
| 1  | 6.31 ± 0.04 (n=2) | 14.26 ± 1.46 (n=2) | 2.26x |
| 4  | 11.36 ± 1.23 (n=2) | **17.23 ± 0.23** (n=4) | 1.52x |
| 16 | 13.78 ± 1.79 (n=4) | 17.68 ± 1.06 (n=4) | 1.28x |
| 64 | 17.74 ± 1.13 (n=4) | 17.28 ± 0.52 (n=4) | 0.97x |

구현 메모:
- 목적지 슬롯도 불연속이라 host gather 만으로는 부족하다. **scatter 커널이 필수**다.
  청크당 CUDA 호출 3회 — 가중치 H2D, 슬롯 인덱스 H2D, scatter 커널.
- 실행기(`SingleThreadExecutor`)가 전역 하나이고 태스크가 직렬화되므로 staging 버퍼도
  전역 하나를 재사용한다. 청크 끝마다 동기화한다.
- DFR 스코어의 뉴런 단위 일반화는 **필요 없었다.** `regroup_model_split.py` 로
  `ffn_group_size` 만 바꾸면 기존 코드가 그대로 뉴런 단위로 동작한다.

### 6-3. 다음 — 약 17.5 tok/s 천장의 정체

gather 를 켜면 gs=4/16/64 가 전부 17.2~17.7 로 수렴한다. PCIe 트래픽이 24.6 GB 와
95.1 GB 로 3.9배 차이나는데도 같다. **따라서 남은 병목은 전송이 아니다.**
후보: sparse matmul 커널, predictor(`ffn_pred_*`), DFR top-k, 또는 정책 자체.
프로파일러(`nsys` / `ncu`)로 커널 단위 분해가 다음 순서다.

부수적으로 gs=1 이 gs=4 보다 느린(14.26 vs 17.23) 원인도 여기 걸려 있다 —
`group_identity` 462 MiB 가 캐시 용량을 잠식하고, top-k 가 11,008 그룹에서 CUB 로
폴백하며, 4,352 B 조각 11,008 개를 모으는 host memcpy 가 바이트당 비효율적이다.
`group_identity` 는 DFR fusion 이 활성이면 읽히지 않으므로 제거 여지가 있다.

### 6-4. 반나절 — 프롬프트 3~5개 반복

여전히 프롬프트 하나다. group_size 별 반복은 2~4회 확보했지만 프롬프트 다양성은 없다.
`PROMPT_FILE=` 로 지정 가능.

⚠️ 런간 변동이 크다(baseline gs=16 에서 11.14~14.98 tok/s). KAIROX 는 reload 가 계산과
비동기로 경쟁해 같은 seed 로도 생성 텍스트가 달라지므로, 단일 런 비교는 신뢰할 수 없다.
**최소 3회 반복이 필요하다.**

### 6-5. 별도 과제 — 예측 정책

`group_size=1`은 뉴런 단위 결정이라 그룹 경계 손실이 원리적으로 없는데도 적중률이
60.99%에 머문다. **granularity는 정확도 병목이 아니며 EMA 정책 자체가 천장이다.**
정적 배치(빈도 상위 N개 고정)는 같은 용량에서 70~85%에 도달한다. 이 격차는 별도로
다뤄야 한다. README 5-5절 참조.

---

## 7. 함정 모음

작업하며 실제로 걸린 것들.

**1024 제약이 두 겹이다.** `llama-kairox.cpp`의 것은 저자의 성능 가이드라인이고
(`ggml_argsort_top_k`는 `ncols > 1024`에서 CUB로 폴백한다), **진짜 제약은
`dfr-fusion.cu`의 공유 메모리 비트마스크**였다. 둘 다 풀어야 한다.

**`--no-mmap`은 뺄 수 없다.** `reorder_if_exists()`가 FFN 가중치를 제자리 순열
재배치하고 reload 경로가 pinned 호스트 버퍼에서 복사한다. 대신 pinned 4.5 GiB가
물리 RAM에 박히므로 **IDE 등 메모리 점유 프로세스를 정리해야 한다.** JetBrains Rider가
3.9 GiB를 잡고 있어 모델 로딩이 반복 실패했다.

**`IGNORE_EOS=1` 없이는 비교가 오염된다.** group_size가 바뀌면 GPU 상주 집합이 바뀌고,
GPU/CPU 경로의 부동소수점 누적 순서 차이가 샘플링을 거쳐 증폭되어 생성 텍스트가
발산한다. 길이는 고정되지만 **텍스트 발산 자체는 남는다** — `P(active)`가 런마다
22.2~24.6%로 변동한다. 엄밀한 통제가 필요하면 활성화 trace를 한 번 기록하고 정책을
오프라인 재생하는 방식이 필요하다.

**계측 함수 호출 순서.** `kairox_dump_residency_counts()`는 반드시
`kairox_reload_plan()` **뒤에** 와야 한다. 스왑 후 `neuron_mask`가 실제로 sparse matmul이
읽는 상태이기 때문이다. 앞에 두면 "지금 필요해서 방금 로드한 뉴런"의 사용이 누락되어
`wasted_loads`가 구조적으로 부풀려진다. (README 2절에 원래 반대로 적혀 있었고 정정함)

**누적 적중과 낭비 판정 플래그를 겸하면 안 된다.** `dbg_hit_count`(리셋 안 함)와
`dbg_used_since_load`(로드 시 리셋)는 별개여야 한다. 하나로 쓰면 적중률이 과소집계된다.

**`total_loads == 0`은 낭비 집계에서 제외한다.** 초기 상주분은 `[0, n_cached_neurons)`
고정 배치이지 DFR의 결정이 아니다. 종료 시점에 상주 중인 뉴런은 evict를 거치지 않으므로
소멸자에서 한 번 훑어 wasted로 계산한다(상한 해석).

**CSV는 소멸자에서 기록된다.** Ctrl+C로 끊으면 파일이 안 남는다.

**`group_identity`는 fusion 시 읽히지 않는다.** `ggml_cuda_op_dfr_mask`는 `topk_idx`만
받는다. 그럼에도 `n_group × n_group` F32로 할당되어 gs=1에서 462 MiB를 차지한다.
8 GiB 카드에서 동작은 확인했으나 제거 여지가 있다.

---


### KAIROX 는 같은 seed 로도 결정적이지 않다

reload 가 계산과 비동기로 경쟁하므로 같은 설정으로 두 번 돌리면 생성 텍스트와 계측 CSV가
서로 다르다. 직접 확인했다(`dumps/gather_ab/ctl_a.csv` vs `ctl_b.csv`).

따라서 **출력 비교로 전송 경로의 정확성을 검증할 수 없다.** gather 구현 때 이것 때문에
처음에 "불일치"가 나와 버그로 오인했다. 정확성은 `KAIROX_GATHER_VERIFY=1` 로 GPU 캐시
슬롯을 되읽어 원본과 `memcmp` 하는 방식으로 검증해야 한다.

같은 이유로 **성능 비교는 반드시 반복해야 한다.** baseline gs=16 에서 11.14~14.98 tok/s 가
관측됐다 — 단일 런 차이는 의미가 없다.

### 처리량 비교에는 `DUMP=0` 이 필수

계측은 레이어마다 D2H 복사 + `cudaStreamSynchronize` 를 추가한다. `dump_activation.sh` 는
기본이 `DUMP=1` 이므로, tok/s 를 비교할 때 끄지 않으면 값이 오염된다.

## 8. 미해결 질문

- `reload_window_size`를 늘리면 정말 2.5배 빨라지는가? (6-1, 미검증)
- gather의 CPU 측 비용을 계산과 중첩시킬 수 있는가? 단일 스레드 8.02 GB/s로 측정됐고
  필요 지속 처리율은 1.22 GB/s라 여유는 있으나 **가정이지 측정이 아니다**
- EMA 대신 어떤 정책이 61% 천장을 넘는가? 정적 배치가 70~85%를 내는 이유는?
- 레이어 0의 wrap-around stale을 구조적으로 고칠 수 있는가?
