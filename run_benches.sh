#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd ./blas_tests
cargo bench --features blas
