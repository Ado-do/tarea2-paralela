#!/bin/bash

set -e

# Configuration
BIN="./bin/benchmark"
RESULTS_EXP1="./data/exp1_benchmark_results.csv"
RESULTS_EXP2="./data/exp2_benchmark_results.csv"

# Configuración de variables
NUM_IMAGES=100
BATCH_SIZES_EXP1=(16 32 64 100)     # Para Experimento 1
BATCH_SIZE_EXP2=8                   # Lote fijo para Experimento 2
STREAMS_EXP2=(1 2 4 8 16)           # Streams a evaluar en Experimento 2
NUM_RUNS=20                         # Iteraciones para validación estadística

echo "[INFO] Comenzando benchmarking..."

# Inicializamos csv
echo "num_images,time_h2d_ms,time_kernels_ms,time_d2h_ms" > $RESULTS_EXP1
echo "num_images,batch_size,num_streams,time_total_ms" > $RESULTS_EXP2

# Nos aseguramos de que esté todo al día
make clean
make all

echo "========================================"
echo "[INFO] EXPERIMENTO 1"
echo "========================================"
for size in "${BATCH_SIZES_EXP1[@]}"; do
    echo "----------------------------------------"
    echo "[INFO] Procesando tamaño de batch: $size"
    
    # Conseguimos métricas de tiempo (NUM_RUNS ejecuciones)
    echo "       -> Recolectando métricas de ejecución ($NUM_RUNS iteraciones)..."
    for i in $(seq 1 $NUM_RUNS); do
        $BIN 1 $NUM_IMAGES $size 1 | grep "\[METRICAS\]," | sed 's/\[METRICAS\],//' >> $RESULTS_EXP1
    done
    
    # Profiling, se corre aparte (1 sola ejecución para visualización)
    echo "       -> Generando nsys trace..."
    nsys profile \
        --trace=cuda,osrt \
        --force-overwrite=true \
        --export=sqlite \
        -o "./data/exp1_profile_batch${size}" \
        $BIN 1 $NUM_IMAGES $size 1 > /dev/null 2>&1

    echo "       -> Trace guardada como ./data/exp1_profile_batch${size}.nsys-rep"
done

echo "========================================"
echo "[INFO] EXPERIMENTO 2"
echo "========================================"
for s in "${STREAMS_EXP2[@]}"; do
    echo "----------------------------------------"
    echo "[INFO] Procesando con $s Streams..."
    
    # Conseguimos métricas de tiempo (NUM_RUNS ejecuciones)
    echo "       -> Recolectando métricas de ejecución ($NUM_RUNS iteraciones)..."
    for i in $(seq 1 $NUM_RUNS); do
        $BIN 2 $NUM_IMAGES $BATCH_SIZE_EXP2 $s | grep "\[METRICAS_EXP2\]," | sed 's/\[METRICAS_EXP2\],//' >> $RESULTS_EXP2
    done
    
    # Profiling, se corre aparte (1 sola ejecución para visualización)
    echo "       -> Generando nsys trace..."
    nsys profile \
        --trace=cuda,osrt \
        --force-overwrite=true \
        --export=sqlite -o "./data/exp2_profile_streams${s}" \
        $BIN 2 $NUM_IMAGES $BATCH_SIZE_EXP2 $s > /dev/null 2>&1
    
    echo "       -> Trace guardada como ./data/exp2_profile_streams${s}.nsys-rep"
done

echo "----------------------------------------"
echo "[INFO] Benchmarking completo. Resultados listos en carpeta ./data"
