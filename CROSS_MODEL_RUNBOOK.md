# 여러 모델에서 group_size 스윕 돌리기

## 왜 하는가

단일 모델(prosparse-7b, n_ff=11008)에서 잰 결과는 **입도가 고울수록 적중률이 오른다**는 것이었다.
캐시 용량을 고정한 채(상주 슬롯-토큰 182.5M 동일) group_size 만 바꿨을 때:

| g | n_group | 적중률 | 미스율 |
|---|---|---|---|
| 2 | 5504 | 80.06% | 19.94% |
| 4 | 2752 | 76.01% | 23.99% |
| 8 | 1376 | 72.35% | 27.65% |
| **16 (배포값)** | **688** | **68.39%** | 31.61% |
| 32 | 344 | 68.81% | 31.19% |
| 64 | 172 | 66.36% | 33.64% |
| 128 | 86 | 65.63% | 34.37% |

배포값 g=16 대비 g=2 는 **+11.67%p, 미스 −37%** 다. 그런데 KAIROX 는 왜 16 을 골랐나?

## 가설 — 고정된 것은 group_size 가 아니라 n_group ≤ 1024 다

논문 8장(Implementation):

> Because high-performance argsort is most efficient on inputs below 1024 elements,
> we constrain the neuron group count M to this threshold.

`bench_models.sh` 의 split 파일명이 이것을 그대로 보여준다. **파일명의 숫자가 n_group 이다.**

| 모델 | 배포 split 의 n_group |
|---|---|
| SparseQwen2-7B | 592 |
| prosparse-llama-2-7b | 688 |
| prosparse-llama-2-13b | 864 |
| Bamboo-base-v0_1 | 896 |
| opt-6.7b | 1024 |
| opt-13b | 1024 |
| opt-30b | 1024 |
| ReluFalcon-40B | 1024 |
| opt-66b | 1024 |

**전부 1024 이하이고, 큰 모델은 정확히 1024 에 붙어 있다.** 즉 KAIROX 가 고정한 것은
group_size 가 아니라 그룹 개수다. group_size 는 `n_ff / n_group` 으로 따라 나올 뿐이다.

n_ff 를 표준 아키텍처 값으로 놓고 역산하면 group_size 는 이렇게 흩어진다:

| 모델 | n_ff (추정) | n_group | 함의된 g |
|---|---|---|---|
| prosparse-llama-2-7b | **11008 (확인됨)** | 688 | **16 (확인됨)** |
| prosparse-llama-2-13b | 13824 | 864 | 16 |
| Bamboo-base-v0_1 | 14336 | 896 | 16 |
| opt-6.7b | 16384 | 1024 | 16 |
| SparseQwen2-7B | 18944 | 592 | 32 |
| opt-13b | 20480 | 1024 | 20 |
| opt-30b | 28672 | 1024 | 28 |
| ReluFalcon-40B | 32768 | 1024 | 32 |
| opt-66b | 36864 | 1024 | 36 |

> **n_group 열만 확실하다** — 파일명에서 직접 읽은 값이다.
> n_ff 와 g 는 prosparse-7b 를 뺀 나머지가 **추정**이다(표준 아키텍처 n_ff 를 가정하고 역산).
> `gs_multi_model.sh` 는 split GGUF 에서 실제 값을 읽으므로 이 표를 믿을 필요는 없다.
> 손으로 확인하려면 아래 "n_ff 확인" 참고.

그러므로 **g=16 은 7B급 모델에서 n_group ≤ 1024 제약이 허용하는 가장 고운 입도**로 보인다.
우리 측정대로라면 그 제약이 **적중률 11.67%p** 를 비용으로 치르고 있고, 제약이 더 세게 걸리는
큰 모델(g=28~36)에서는 손실이 더 클 것으로 예상된다.

## 여러 모델이 필요한 이유 — 세 결과가 다른 결론을 낳는다

모델별 최적 g(적중률 기준)를 재면 셋 중 하나가 나온다.

1. **최적 g 가 모델 무관하게 일정 (≈2)**
   → 입도는 워크로드가 정하지 않는다. 배포값은 순수하게 argsort 제약의 산물이고,
   제약을 푸는 것 자체가 기여가 된다.
2. **최적 n_group 이 대체로 일정 (최적 g 가 n_ff 에 비례)**
   → 본질은 뉴런 수가 아니라 그룹 개수다. 모델마다 g 를 달리 잡아야 한다는 뜻이 된다.
3. **최적 g 가 배포값과 같다**
   → 논문 선택이 워크로드로도 정당화된다. 우리 결과가 prosparse 고유 현상이므로
   주장을 접거나 범위를 좁혀야 한다.

세 경우가 깨끗이 구분되므로 모델 수가 늘수록 결론이 강해진다. 특히 **큰 모델(g=28~36)이
중요하다** — 제약이 가장 세게 걸린 지점이라 1번과 2번을 가장 잘 가른다.

## 준비물

### 1. 코드

이 브랜치(`feat/anb`)를 그대로 가져간다.

- `gs_multi_model.sh` — 이 작업의 본체. 모델 목록이 하드코딩돼 있고 없는 건 건너뛴다
- `dump_activation.sh` — 러너. ANB / τ / gather 노브를 전부 환경변수로 받는다
- `regroup_model_split.py` — split GGUF 의 group_size 만 바꿔 재작성. **순열은 그대로 두고
  더 잘게 끊어 읽기만 하므로 클러스터링을 다시 돌릴 필요가 없다**

```sh
git clone <이 저장소> && cd kairox.cpp && git checkout feat/anb
bash compile_kairox.sh rel
```

### 2. 모델

모델마다 **본 모델 GGUF 1개 + 배포된 model-split GGUF 1개**면 된다.
나머지 group_size 의 split 은 스크립트가 만들어 쓴다.

```sh
hf download Anhelor/SPIF-GGUF --repo-type model --local-dir "$HOME/SPIF-GGUF"
```

### 3. n_ff 확인 (선택)

```sh
python3 - "$HOME/SPIF-GGUF/<모델>-sparkinfer-model-split-<N>.gguf" <<'PY'
import sys; sys.path.insert(0, "gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); f = r.fields["ffn_group_size"]
n_ff, gs = len(r.tensors[0].data), int(f.parts[f.data[0]][0])
print(f"n_ff={n_ff} group_size={gs} n_group={n_ff//gs}")
PY
```

## 돌리는 법

```sh
bash gs_multi_model.sh                                  # 보유한 모델 전부
MODELS="opt-6.7b opt-13b" bash gs_multi_model.sh        # 일부만
SIZES="4 16 64" BENCH_RUNS=1 bash gs_multi_model.sh     # 축 좁혀 경향만
```

출력은 `gs_multi_logs/<모델명>/gs*.csv` 와, 모델별 표 + 마지막의 교차 비교표다.

| 열 | 뜻 |
|---|---|
| 적중률 | 활성 뉴런 중 GPU 상주였던 비율 — **주 지표** |
| 미스율 | 1 − 적중률. CPU 로 넘어가는 연산량에 비례 |
| 로드 | 총 로드 뉴런 수 |
| 시간 | 고정 토큰 수 디코드에 걸린 초. 낮을수록 좋음 |

## 반드시 손볼 것 — vb (용량 축)

**모델 간 비교에서 가장 깨지기 쉬운 지점이다.** 적중률은 입도보다 **캐시 용량이 훨씬 강하게
지배한다** (같은 모델에서 vb 5/6/7 → 18.3 / 53.4 / 70.4%). 모델마다 n_ff 와 가중치 크기가
다르므로 같은 vb 를 줘도 상주 비율이 제각각이 되고, 그러면 입도 효과가 용량 효과에 묻힌다.

스크립트 상단 `ENTRIES` 배열의 세 번째 필드가 모델별 vb 다. **12 GiB 기준 어림값이므로
쓰는 GPU 에 맞게 고쳐야 한다.**

```
    "opt-13b                   | opt-13b-sparkinfer-model-split-1024.gguf              | 8"
                                                                                         ^^ 여기
```

맞추는 기준은 **상주 뉴런 비율**이다. 로그의 이 줄로 확인한다:

```
kairox_init_...: [layer  0] offloaded  xxx MiB and cached  5187 ( 47.12%) neurons to device
```

현재 단일 모델 결과는 상주 비율 ≈ 47% 에서 얻었으므로, 모델마다 vb 를 조정해 이 값을
비슷하게 맞추면 직접 비교할 수 있다. 한 모델당 짧은 런(`SIZES="16" BENCH_RUNS=1 N=32`)을
한 번 돌려 비율을 보고 vb 를 정한 뒤 본 스윕을 돌리는 것이 빠르다.

## 정책 조건

기본값은 **"그루핑된 TAM 정책" 격리 조건**이다. 논문에 없는 스왑 예산(`dfr_clamp_k`)을 끄고,
one-hit wonder 필터 τ 를 실측 최적값 근처(≈0)에 둔다. 둘 다 입도와 얽혀 곡선을 왜곡한다.

| 조건 | 설정 |
|---|---|
| 격리 (기본) | `ANB=1 LAMBDA_MIN=0.67 LAMBDA_MAX=0.67 TAU_LOAD=0.0001` |
| as-shipped | `ANB=0 TAU_LOAD=0.33 OUT_DIR=./as_shipped` |

두 조건을 모두 돌려두면 "논문 기본 설정에서는 이 효과가 안 보이고, 분리하면 보인다"를
같은 데이터로 보일 수 있다.

τ 가 입도와 상호작용하지 않는다는 것은 prosparse-7b 에서 확인했다 — τ=0.33 과 τ≈0 의 기울기가
g=8→128 구간에서 −6.55 vs −6.72 %p 로 사실상 같다. 다른 모델에서도 성립하는지는
`TAU_LOAD` 두 값으로 각각 돌려 확인하면 된다.

## 주의

- **GPU 를 독점해야 한다.** 다른 실행과 겹치면 시간 측정이 통째로 오염된다
  (실제로 겹쳐 재서 7.96 t/s 가 5.85 t/s 로 나온 적이 있다). 적중률 같은 카운트 지표는 무사하다.
- **`--ignore-eos` 로 토큰 수를 고정한다** (스크립트가 자동으로 켠다). 안 그러면 생성 길이가
  달라져 카운터 비교가 오염된다.
- **`BENCH_RUNS=2` 는 경향 확인용이다.** 결론에 넣을 숫자는 5 이상으로 다시 받는다.
  적중률은 카운트 기반이라 안정적이지만 시간은 노이즈가 크다(이 저장소 기준 ±45%).
- **g=1 은 피한다.** `group_identity` 가 n_ff² × 4 바이트라 n_ff=11008 이면 462 MiB 다.
- **호스트 RAM 이 모델보다 빠듯하면 연속 실행이 swap 으로 무너진다** (`--no-mmap` 이라
  모델이 통째로 올라간다). 같은 설정에서 15.5 t/s 와 0.6 t/s 가 섞여 나온 적이 있다.
- CSV 는 정상 종료 시 소멸자에서 쓰인다. `Ctrl+C` 로 끊으면 그 조합은 파일이 남지 않는다.
