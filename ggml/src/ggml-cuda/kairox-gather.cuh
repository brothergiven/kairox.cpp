#pragma once

#include "common.cuh"
#include "ggml-kairox.hpp"

/**
 * 흩어진 그룹들을 pinned staging 버퍼로 모아 H2D 1회 + scatter 커널 1회로 GPU 캐시에 반영한다.
 *
 * 기존 kairox_batch_reload 는 그룹당 cudaMemcpyAsync 를 한 번씩 호출하고 reload_window_size 마다
 * 동기화한다. 이 함수는 그 호출 횟수를 reload_count 개에서 (청크당) 3개로 줄인다.
 *
 * 목적지 슬롯도 흩어져 있으므로 host gather 만으로는 부족하다. staging 버퍼는 연속이지만
 * cache_base + slot_idx * group_nbytes 는 불연속이라, 디바이스 쪽에서 scatter 커널이 필요하다.
 *
 * 실행기(SingleThreadExecutor)가 전역에 하나뿐이고 태스크가 직렬화되므로 staging 버퍼도 전역
 * 하나를 재사용한다. 청크 끝마다 동기화하여 host 버퍼 재사용 전에 전송이 끝난 것을 보장한다.
 */
void kairox_gather_reload(char *              weight_base,
                          char *              cache_base,
                          size_t              group_nbytes,
                          cudaStream_t        stream,
                          const reload_pair * reload_plan,
                          size_t              reload_count);
