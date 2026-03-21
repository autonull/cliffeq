"""
Comprehensive Phase 2 & Phase 3 Tier 1 Re-validation Runner
============================================================

After critical geometric_product bug fix, re-run all Phase 2 (P2.5-P2.10)
and Phase 3 Tier 1 (PG1, PV1) experiments with corrected implementation.

Captures:
- Experiment name, status, runtime
- Key metrics from stdout
- Pass/fail for each experiment
- Generates corrected results report
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Define all experiments to re-validate
PHASE2_EXPERIMENTS = [
    ("P2.5: Clifford-ISTA", "experiments/p2_5_ista.py"),
    ("P2.6: Predictive Coding", "experiments/p2_6_predictive_coding.py"),
    ("P2.7: Target Propagation", "experiments/p2_7_target_propagation.py"),
    ("P2.8: Geometric Attention", "experiments/p2_8_geometric_attention.py"),
    ("P2.9: Bottleneck (V2)", "experiments/p2_9_bottleneck_test_v2.py"),
    ("P2.10: Multi-Algorithm", "experiments/p2_10_multi_algorithm_nbody.py"),
]

PHASE3_EXPERIMENTS = [
    ("PG1: N-Body Baseline", "experiments/p3_1_nbody_dynamics.py"),
    ("PG1: Clifford-EP Variant", "experiments/p3_1_clifford_ep_nbody.py"),
    ("PV1: CIFAR-10 Baseline", "experiments/p3_2_cifar10_rotation.py"),
    ("PV1: Clifford-EP Variant", "experiments/p3_2_clifford_ep_vision.py"),
]

INTEGRATION_EXPERIMENTS = [
    ("P2.9 on Phase 3 (PG1)", "experiments/p2_9_on_phase3_pg1.py"),
]

def run_experiment(exp_name, exp_file, timeout=600):
    """Run a single experiment and capture results."""
    print(f"\n  Running: {exp_name}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, exp_file],
            env={"PYTHONPATH": "/home/me/cliffeq", **dict(subprocess.os.environ)},
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd="/home/me/cliffeq"
        )

        elapsed = time.time() - start_time

        status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
        print(f"    {status} ({elapsed:.1f}s)")

        return {
            "name": exp_name,
            "file": exp_file,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT ({timeout}s)")
        return {
            "name": exp_name,
            "file": exp_file,
            "status": "TIMEOUT",
            "elapsed": timeout,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "returncode": -1
        }

    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        return {
            "name": exp_name,
            "file": exp_file,
            "status": "ERROR",
            "elapsed": 0,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

def main():
    print("=" * 80)
    print("PHASE 2 & PHASE 3 TIER 1 RE-VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    print("\nAfter critical geometric_product bug fix.")
    print("Systematically re-running all affected experiments.\n")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "phase2": [],
        "phase3": [],
        "integration": [],
        "summary": {}
    }

    # Phase 2 experiments
    print("\n" + "=" * 80)
    print("PHASE 2: STRUCTURAL EXPLORATIONS (P2.5-P2.10)")
    print("=" * 80)
    for exp_name, exp_file in PHASE2_EXPERIMENTS:
        result = run_experiment(exp_name, exp_file, timeout=600)
        all_results["phase2"].append(result)

    # Phase 3 Tier 1 experiments
    print("\n" + "=" * 80)
    print("PHASE 3 TIER 1: DOMAIN BENCHMARKS (PG1, PV1)")
    print("=" * 80)
    for exp_name, exp_file in PHASE3_EXPERIMENTS:
        result = run_experiment(exp_name, exp_file, timeout=900)
        all_results["phase3"].append(result)

    # Integration experiments
    print("\n" + "=" * 80)
    print("INTEGRATION: P2.9 ON PHASE 3")
    print("=" * 80)
    for exp_name, exp_file in INTEGRATION_EXPERIMENTS:
        result = run_experiment(exp_name, exp_file, timeout=600)
        all_results["integration"].append(result)

    # Summary
    print("\n" + "=" * 80)
    print("RE-VALIDATION SUMMARY")
    print("=" * 80)

    p2_pass = sum(1 for r in all_results["phase2"] if r["status"] == "PASS")
    p3_pass = sum(1 for r in all_results["phase3"] if r["status"] == "PASS")
    integ_pass = sum(1 for r in all_results["integration"] if r["status"] == "PASS")

    print(f"\nPhase 2 (P2.5-P2.10):       {p2_pass}/{len(PHASE2_EXPERIMENTS)} PASS")
    print(f"Phase 3 Tier 1 (PG1, PV1): {p3_pass}/{len(PHASE3_EXPERIMENTS)} PASS")
    print(f"Integration (P2.9 on P3):  {integ_pass}/{len(INTEGRATION_EXPERIMENTS)} PASS")

    all_results["summary"] = {
        "phase2": f"{p2_pass}/{len(PHASE2_EXPERIMENTS)}",
        "phase3": f"{p3_pass}/{len(PHASE3_EXPERIMENTS)}",
        "integration": f"{integ_pass}/{len(INTEGRATION_EXPERIMENTS)}",
        "total_pass": p2_pass + p3_pass + integ_pass,
        "total": len(PHASE2_EXPERIMENTS) + len(PHASE3_EXPERIMENTS) + len(INTEGRATION_EXPERIMENTS)
    }

    # Save results to JSON
    results_file = Path("/home/me/cliffeq/REVALIDATION_RESULTS.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Print failed experiments
    failed = [r for r in all_results["phase2"] + all_results["phase3"] + all_results["integration"]
              if r["status"] != "PASS"]
    if failed:
        print("\n" + "=" * 80)
        print("FAILED EXPERIMENTS")
        print("=" * 80)
        for r in failed:
            print(f"\n{r['name']} ({r['file']})")
            print(f"Status: {r['status']}")
            if r["stderr"]:
                print(f"Error:\n{r['stderr'][:500]}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review REVALIDATION_RESULTS.json for detailed results")
    print("2. For any failures, check experiment output in JSON file")
    print("3. Generate corrected results report once all pass")
    print("4. Compare against original documented results")
    print("=" * 80 + "\n")

    return 0 if all_results["summary"]["total_pass"] == all_results["summary"]["total"] else 1

if __name__ == "__main__":
    sys.exit(main())
