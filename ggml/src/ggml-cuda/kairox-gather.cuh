#pragma once

#include "common.cuh"
#include "ggml-kairox.hpp"

/**
 * 흩어진 그룹들을 pinned staging 버퍼로 모아 H2D 1회 + scatter 커널 1회로 GPU 캐시에 반영한다.
 *
 * 원본 kairox_batch_reload 는 그룹당 cudaMemcpyAsync 를 한 번씩 호출하고 reload_window_size 마다
 * 동기화한다. 이 함수는 호출 수를 reload_count 개에서 (청크당) 3개로 줄인다.
 *
 * 목적지 슬롯도 흩어져 있으므로 host gather 만으로는 부족하다. staging 버퍼는 연속이지만
 * cache_base + slot_idx * group_nbytes 는 불연속이라 디바이스 쪽 scatter 커널이 필요하다.
 *
 * 실행기(SingleThreadExecutor)가 하나뿐이고 태스크가 직렬화되므로 staging 버퍼는 전역 하나를 재사용한다.
 */
void kairox_gather_reload(char *              weight_base,
                          char *              cache_base,
                          size_t              group_nbytes,
                          cudaStream_t        stream,
                          const reload_pair * reload_plan,
                          size_t              reload_count);


/**
 * zero-copy 경로. 시그니처는 kairox_gather_reload 와 같다.
 * weight_base 가 pinned(디바이스에서 접근 가능) 가 아니면 kairox_gather_reload 로 폴백한다.
 */
void kairox_zerocopy_reload(char *              weight_base,
                            char *              cache_base,
                            size_t              group_nbytes,
                            cudaStream_t        stream,
                            const reload_pair * reload_plan,
                            size_t              reload_count);

/**
 * load/evict 마스크를 인덱스 목록으로 압축한다 (KAIROX_GPU_COMPACT).
 *
 * 원래는 호스트가 마스크 두 개를 D2H 받아 n_groups 를 전부 훑어 목록을 만들었다. 그 스캔이
 * O(n_groups) 라 입도를 잘게 할수록 커진다 — g=16 에서 0.09, g=1 에서 1.35 ms/토큰.
 * 그리고 plan 이 늦어지면 전송이 연산 창을 놓쳐 anchor 에서 막히므로, 호스트 지연이
 * 9~13 배로 증폭된다(실측). 압축을 GPU 로 옮기면 호스트는 실제 짝 개수만큼만 순회한다.
 *
 * 반환 버퍼(pinned host)의 구조:
 *   [0] = n_load, [1] = n_evict, [2 .. 2+n) = load 인덱스, [2+n .. 2+2n) = evict 인덱스
 *
 * atomicAdd 로 모으므로 인덱스 순서는 보장하지 않는다. 예산 절단이 없으면 결과가 같고,
 * 절단이 있으면 어느 그룹이 잘리는지가 달라진다(원본의 번호 순 편향도 없어진다).
 */
const int * kairox_compact_masks(const float * load_mask,
                                 const float * evict_mask,
                                 int           n_groups,
                                 cudaStream_t  stream);