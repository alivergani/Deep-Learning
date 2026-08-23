#!/bin/bash
cd ~/Deep-Learning/ExoticParticles_HEP
mkdir -p logs

for config in "shallow high" "shallow low" "shallow complete" \
              "deep high" "deep low" "deep complete"; do

    nome=$(echo $config | tr ' ' '_')
    echo "=== $nome  $(date +%H:%M) ==="

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 0,2 \
        python src/esperimenti.py $config 0 1   > logs/${nome}_a.log 2>&1 &
    p1=$!
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 4,6 \
        python src/esperimenti.py $config 2 3 4 > logs/${nome}_b.log 2>&1 &
    p2=$!

    wait $p1 $p2

    python src/esperimenti.py $config > logs/${nome}_riepilogo.log 2>&1
done
