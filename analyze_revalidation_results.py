"""
Parse and analyze re-validation results.
Compares original (buggy) vs. corrected results.
Generates summary report.
"""

import json
import re
from pathlib import Path
from datetime import datetime

def extract_metrics(text):
    """Extract key metrics from experiment output."""
    metrics = {}

    # Common metric patterns
    patterns = {
        "accuracy": r"(?:accuracy|Accuracy)[:\s]+([0-9.]+)%?",
        "loss": r"(?:loss|Loss)[:\s]+([0-9.]+[eE]?[-+]?[0-9]*)",
        "mse": r"(?:MSE|mse)[:\s]+([0-9.]+[eE]?[-+]?[0-9]*)",
        "sparsity": r"(?:sparsity|Sparsity)[:\s]+([0-9.]+)%?",
        "equivariance": r"(?:equivariance|Equivariance)[:\s]+([0-9.]+)%?",
        "violation": r"(?:violation|Violation)[:\s]+([0-9.]+)%?",
        "improvement": r"(?:improvement|Improvement)[:\s]+([0-9.]+)%?",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics

def generate_report():
    """Generate re-validation report."""
    results_file = Path("/home/me/cliffeq/REVALIDATION_RESULTS.json")

    if not results_file.exists():
        print("❌ REVALIDATION_RESULTS.json not found. Experiments still running?")
        return

    with open(results_file) as f:
        results = json.load(f)

    print("\n" + "=" * 80)
    print("RE-VALIDATION ANALYSIS REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 80)

    # Summary
    print("\n## SUMMARY\n")
    print(f"Phase 2 (P2.5-P2.10):       {results['summary']['phase2']} passed")
    print(f"Phase 3 Tier 1 (PG1, PV1): {results['summary']['phase3']} passed")
    print(f"Integration (P2.9 on P3):  {results['summary']['integration']} passed")
    print(f"Total:                      {results['summary']['total_pass']}/{results['summary']['total']} passed")

    # Detailed results
    print("\n## PHASE 2 RESULTS\n")
    for result in results["phase2"]:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"{status_icon} {result['name']:40} [{result['status']:6}] ({result['elapsed']:.1f}s)")
        if result["status"] == "PASS":
            metrics = extract_metrics(result["stdout"])
            if metrics:
                for k, v in metrics.items():
                    print(f"    {k:15} {v:.4f}")

    print("\n## PHASE 3 RESULTS\n")
    for result in results["phase3"]:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"{status_icon} {result['name']:40} [{result['status']:6}] ({result['elapsed']:.1f}s)")
        if result["status"] == "PASS":
            metrics = extract_metrics(result["stdout"])
            if metrics:
                for k, v in metrics.items():
                    print(f"    {k:15} {v:.4f}")

    print("\n## INTEGRATION RESULTS\n")
    for result in results["integration"]:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"{status_icon} {result['name']:40} [{result['status']:6}] ({result['elapsed']:.1f}s)")
        if result["status"] == "PASS":
            metrics = extract_metrics(result["stdout"])
            if metrics:
                for k, v in metrics.items():
                    print(f"    {k:15} {v:.4f}")

    # Failed experiments
    failed = [r for r in results["phase2"] + results["phase3"] + results["integration"]
              if r["status"] != "PASS"]
    if failed:
        print("\n## FAILED EXPERIMENTS\n")
        for result in failed:
            print(f"✗ {result['name']}")
            print(f"  Status: {result['status']}")
            if result["stderr"]:
                stderr_lines = result["stderr"].split("\n")[:5]
                for line in stderr_lines:
                    if line:
                        print(f"    {line[:70]}")
            print()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review failures above (if any)")
    print("2. Create REVALIDATION_CORRECTED_RESULTS.md with all metrics")
    print("3. Compare against original PHASE2_RESULTS_REPORT.md")
    print("4. Mark results as 'VALIDATED' ready for Phase 4")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_report()
