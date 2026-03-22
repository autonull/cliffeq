import subprocess
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

def run_experiment(script_name, experiment_name):
    print(f"\nStarting: {experiment_name}")
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, f'experiments/{script_name}'],
            capture_output=True, text=True, timeout=3600, cwd=project_root,
            env=env
        )
        print(result.stdout)
        if result.stderr: print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {experiment_name} failed: {e}")
        return False

def main():
    print("PHASE 4: CLIFFORD ABLATION STUDY (EP VS BP)")
    scripts = [
        ('p4_1_resnet_clifford_bottleneck.py', 'Vision'),
        ('p4_2_transformer_sentiment.py', 'Language'),
        ('p4_3_ppo_cartpole.py', 'RL'),
        ('p4_4_gnn_graph_classification.py', 'Graphs')
    ]
    for script, name in scripts:
        run_experiment(script, name)
    print("\n✓ ORCHESTRATION COMPLETE")

if __name__ == "__main__": main()
