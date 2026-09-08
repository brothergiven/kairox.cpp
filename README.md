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
| `TOKEN_LATENCY` | `0` | `1`이면 per-token latency 패스를 조합마다 한 번 더 돈다 (4-4 참고) |
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
| `TOKEN_LATENCY` / `TPOT_OUT` | `0` / `<OUT>_tpot.csv` | TPOT 패스로 전환 (4-4 참고) |

프롬프트는 `--bench-prompt-file`로 넘어가고, 계측 카운터는 런 사이에 리셋되지 않으므로 CSV는
**여러 프롬프트에 걸친 합계**가 된다. 워밍업 런도 카운터에 누적되기 때문에 `--bench-warmup 0`을
쓴다.

### 4-4. TPOT (per-token decode latency)

**평균 TPOT는 항상 기록된다.** `decode mean`의 역수이므로 추가 실행 없이 요약표와
`*_summary.csv`에 `tpot_ms` 열로 들어간다.

```text
TPOT(ms/token) = 1000 / decode mean(t/s)
```

분포까지 보려면 `TOKEN_LATENCY=1`을 준다. 조합마다 실행을 한 번 더 해서 토큰별 지연을
받아 p50/p90/p99/max를 낸다. wasted rebalancing으로 생기는 stall은 평균보다 꼬리에서
드러나므로 이쪽이 더 직접적인 지표다.

```bash
TOKEN_LATENCY=1 BACKENDS="kairox" VBS="6" bash bench_group_sweep.sh full
```

```text
TPOT 분포 (ms/token, per-token 측정)
backend      vb     gs        n       mean        p50        p90        p99
kairox        6      8      511      88.06      84.66      88.98     250.45
```

결과는 조합별 `gs<N>_tpot.csv`(`token_index,latency_ms`)와 `tpot_summary.csv`에 남는다.

별도 패스인 이유는 `llama-completion`의 제약 때문이다. `--bench-token-latency`는
`--bench-prompt-file`과 같이 쓸 수 없고 `--bench-runs 1`만 받는다. 그래서 이 패스는
프롬프트 하나(`PROMPT` 또는 `PROMPT_FILE`의 첫 줄)로 워밍업 1 + 측정 1런을 돈다.
activation 덤프는 꺼진 채로 돌기 때문에 본 측정 CSV를 덮어쓰지 않는다. 대신 프롬프트
구성이 본 측정과 다르므로, 평균 TPOT 열과 분포 표의 값은 정확히 일치하지 않는다.

### 4-5. 재개와 재측정

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

### 4-6. 주의

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
