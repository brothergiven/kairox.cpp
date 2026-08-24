#!/bin/bash
# sweep the shared VRAM budget to locate the memory-constrained regime on an 8 GB card
cd /workspace/kairox.cpp || exit 1
M=/root/SPIF-GGUF/prosparse-llama-2-7b-Q8_0.gguf
MS=/root/SPIF-GGUF/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf
OUT=vb_sweep_logs
mkdir -p "$OUT"
for vb in 5 6 7; do
    for backend in llama_cpp kairox neuralink; do
        log="$OUT/e2e_performance__${backend}__3070vb${vb}__completion__prosparse-llama-2-7b-Q8_0.log"
        [[ -s "$log" ]] && { echo "skip  $log"; continue; }
        args=("$backend" 3070 kind=completion "vb=$vb" "model=$M" bench bench_runs=10)
        [[ "$backend" != llama_cpp ]] && args+=("model_split=$MS")
        bash test_kairox.sh "${args[@]}" >"$log" 2>&1
        printf '%-10s vb=%s -> %s\n' "$backend" "$vb" \
            "$(grep -m1 'decode mean' "$log" || echo 'NO SUMMARY')"
    done
done
