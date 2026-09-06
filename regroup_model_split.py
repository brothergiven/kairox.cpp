#!/usr/bin/env python3
"""
model-split GGUF 의 ffn_group_size 만 바꿔서 재작성한다.

DFR 의 결정 단위(group_size)를 좁혔을 때 캐시 적중률이 얼마나 개선되는지 재기 위한
도구다. ffn_reorder_perms 는 group_size 와 무관한 순수 뉴런 재배치이므로, 같은 순열을
더 잘게 끊어 읽기만 하면 된다 — 클러스터링을 다시 돌릴 필요가 없다.

  python3 regroup_model_split.py <원본.gguf> <group_size> [출력.gguf]

주의: llama-kairox.cpp 는 group_identity 를 n_group x n_group (F32) 로 GPU 에 올린다.
n_group = n_ff / group_size 이므로 이 텐서는 group_size 에 제곱 반비례로 커진다.
11008 뉴런 기준 group_size=4 는 30 MB, 2 는 121 MB, 1 은 485 MB 다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))

import numpy as np
from gguf import GGUFReader, GGUFWriter


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__.strip(), file=sys.stderr)
        return 1

    src_path = Path(sys.argv[1])
    group_size = int(sys.argv[2])

    if not src_path.is_file():
        print(f"error: 원본 파일 없음: {src_path}", file=sys.stderr)
        return 1
    if group_size < 1:
        print(f"error: group_size 는 1 이상이어야 한다: {group_size}", file=sys.stderr)
        return 1

    reader = GGUFReader(str(src_path))

    def scalar(key):
        field = reader.fields[key]
        return field.parts[field.data[0]][0]

    def string(key):
        field = reader.fields[key]
        return bytes(field.parts[field.data[0]]).decode()

    arch = string("general.architecture")
    orig_group_size = int(scalar("ffn_group_size"))

    pattern_field = reader.fields["ffn_normalized_pattern"]
    pattern = [float(pattern_field.parts[i][0]) for i in pattern_field.data]

    perms = [(t.name, np.array(t.data, dtype=np.int32)) for t in reader.tensors]
    n_neurons = len(perms[0][1])

    if n_neurons % group_size:
        print(
            f"error: n_ff({n_neurons}) 가 group_size({group_size}) 로 나눠떨어지지 않는다",
            file=sys.stderr,
        )
        return 1

    n_group = n_neurons // group_size
    identity_mib = n_group * n_group * 4 / (1024 * 1024)

    if len(sys.argv) == 4:
        dst_path = Path(sys.argv[3])
    else:
        # 원본 파일명 끝의 그룹 개수를 새 값으로 교체한다 (...-split-688.gguf -> ...-split-2752.gguf)
        stem = src_path.stem
        old_suffix = f"-{n_neurons // orig_group_size}"
        stem = stem[: -len(old_suffix)] if stem.endswith(old_suffix) else stem
        dst_path = src_path.with_name(f"{stem}-{n_group}.gguf")

    print(f"원본        : {src_path}")
    print(f"group_size  : {orig_group_size} -> {group_size}")
    print(f"n_group     : {n_neurons // orig_group_size} -> {n_group}")
    print(f"뉴런 수     : {n_neurons}")
    print(f"레이어 수   : {len(perms)}")
    print(f"group_identity 예상 크기 : {identity_mib:.1f} MiB (GPU)")
    if identity_mib > 128:
        print("  경고: group_identity 가 매우 크다. VRAM 여유를 확인할 것.")
    print(f"출력        : {dst_path}")

    writer = GGUFWriter(str(dst_path), arch)
    writer.add_int32("ffn_group_size", group_size)
    writer.add_array("ffn_normalized_pattern", pattern)
    for name, data in perms:
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"완료: {dst_path.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
