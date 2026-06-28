#!/bin/bash

set -e

# Configuration
BIN="./bin/benchmark"
RESULTS_FILE="./data/exp1_benchmark_results.csv"
BATCH_SIZES=(16 32 64 100) # Máximo 100
NUM_RUNS=20 # Iteraciones para validación estadística

echo "[INFO] Comenzando benchmarking..."

# Inicializamos csv
echo "batch_size,time_h2d_ms,time_kernels_ms,time_d2h_ms" > $RESULTS_FILE

# Nos aseguramos de que esté todo al día
make clean
make all

for size in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------"
    echo "[INFO] Procesando tamaño de batch: $size"

    # Conseguimos métricas de tiempo (NUM_RUNS ejecuciones)
    echo "       -> Recolectando métricas de ejecución ($NUM_RUNS iteraciones)..."
    for i in $(seq 1 $NUM_RUNS); do
        $BIN $size | grep "\[METRICAS\]" | sed 's/\[METRICAS\],//' >> $RESULTS_FILE
    done

    # Profiling, se corre aparte (1 sola ejecución para visualización)
    echo "       -> Generando nsys trace..."
    nsys profile \
        --trace=cuda,osrt \
        --force-overwrite=true \
        --export=sqlite \
        -o "./data/exp1_profile_batch${size}" \
        $BIN $size > /dev/null 2>&1

    echo "       -> Trace guardada como ./data/exp1_profile_batch${size}.nsys-rep"
done

echo "----------------------------------------"
echo "[INFO] Benchmarking completo. Resultados listos para Python en $RESULTS_FILE"
