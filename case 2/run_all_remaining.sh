#!/bin/bash
for name in T01_slow_r1 T02_slow_r1 T04_slow_r1 T05_slow_r1 T06_slow_r1 T07_slow_r1 T09_slow_r1 T10_slow_r1 T11_slow_r1 "reach_extend_v0.6_a1.5" "short_moves_v0.6_a1.5" "wrist_sweep_v0.6_a1.5"; do
  out="results/hardware_tests/$name"
  if [ -f "$out/baseline_result.csv" ] && [ -f "$out/optimized_result.csv" ]; then
    echo "skipping $name -- already have results"
    continue
  fi
  echo "=== $name ==="
  python3 run_ab_test.py --folder "manual_scripts/$name" --robot-ip 192.168.1.100
done
