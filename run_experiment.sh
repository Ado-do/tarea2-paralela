#!/bin/bash

set -e

# Configuration
BIN="./bin/benchmark"
RESULTS_FILE="./data/benchmark_results.csv"
BATCH_SIZES=(16 32 64 100) # Máximo 100

echo "[INFO] Comenzando benchmarking..."

# Inicializamos csv
echo "batch_size,time_h2d_ms,time_kernels_ms,time_d2h_ms" > $RESULTS_FILE

# Nos aseguramos de que esté todo al día
make clean
make all

for size in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------"
    echo "[INFO] Procesando tamaño de batch: $size"

    # Conseguimos métricas de tiempo
    echo "       -> Recolectando métricas de ejecución..."
    $BIN $size | grep "\[METRICAS\]" | sed 's/\[METRICAS\],//' >> $RESULTS_FILE

    # Profiling, se corre aparte
    echo "       -> Generando nsys trace..."
    # Run nsys quietly, dumping the artifact to the workspace
    nsys profile \
        --trace=cuda,osrt \
        --force-overwrite=true \
        --export=sqlite \
        -o "profile_exp1_batch${size}" \
        $BIN $size > /dev/null 2>&1

    echo "       -> Trace guardada como profile_exp1_batch${size}.nsys-rep"
done

echo "----------------------------------------"
echo "[INFO] Benchmarking completo. Resultados en $RESULTS_FILE"
