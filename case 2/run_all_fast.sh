#!/bin/bash
for name in T01_fast_r1 T02_fast_r1 T03_fast_r1 T04_fast_r1 T05_fast_r1 T06_fast_r1 T07_fast_r1 T08_fast_r1 T09_fast_r1 T10_fast_r1 T11_fast_r1 "reach_extend_v1.8_a3" "short_moves_v1.8_a3" "workspace_sweep_v1.8_a3" "wrist_sweep_v1.8_a3"; do
  out="results/hardware_tests/$name"
  if [ -f "$out/baseline_result.csv" ] && [ -f "$out/optimized_result.csv" ]; then
    echo "skipping $name -- already have results"
    continue
  fi
  echo "=== $name ==="
  python3 run_ab_test.py --folder "manual_scripts/$name" --robot-ip 192.168.1.100
done
