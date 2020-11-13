#!/bin/bash

cd ./blas_tests
cargo test $*
cargo test --features blas $*
