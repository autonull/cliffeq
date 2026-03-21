"""Quick Phase 2 experiment runner - collects results without full dependencies"""
import subprocess
import sys
import json
from pathlib import Path

results = {}
experiments = [
    ("p2_5_ista", "experiments/p2_5_ista.py"),
    ("p2_8_attention", "experiments/p2_8_geometric_attention.py"),
    ("p2_9_bottleneck", "experiments/p2_9_bottleneck_test.py"),
    ("p2_10_nbody", "experiments/p2_10_multi_algorithm_nbody.py"),
]

print("=" * 70)
print("PHASE 2 QUICK EXPERIMENT RUN")
print("=" * 70)

for exp_name, exp_file in experiments:
    print(f"\n[{exp_name}] Running...")
    result = subprocess.run(
        ["python", exp_file],
        env={"PYTHONPATH": "/home/me/cliffeq", **dict(subprocess.os.environ)},
        capture_output=True,
        timeout=120,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✓ {exp_name} completed successfully")
        results[exp_name] = "SUCCESS"
    else:
        print(f"  ✗ {exp_name} failed")
        results[exp_name] = "FAILED"
        if "Reconstruction accuracy:" in result.stdout:
            results[exp_name] += " (partial)"

print("\n" + "=" * 70)
print("PHASE 2 RESULTS SUMMARY")
print("=" * 70)
for exp, status in results.items():
    print(f"{exp:30} {status}")
