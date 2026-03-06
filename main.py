"""
Attention / Behaviour Detection Model — Main Entry Point.

Usage:
    python main.py extract      — Extract V4 generalised features (MediaPipe + YOLO + phone-aware)
    python main.py train        — Train XGBoost + RF on V4 features
    python main.py live         — Run live webcam V5 (generalised + phone detection)  ★
    python main.py extract-v3   — Extract V3 features (legacy)
    python main.py train-v3     — Train on V3 features (legacy)
    python main.py live-v3      — Run live V3 (legacy)
    python main.py live-rules   — Run live V4 rule-based engine (legacy)
    python main.py all          — Extract → Train → Live (V4/V5 pipeline)
    python main.py all-v3       — Extract → Train → Live (V3 legacy pipeline)
"""

import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")


def run_script(name):
    path = os.path.join(SCRIPTS_DIR, name)
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, path], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\nERROR: {name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    # ── V4/V5 (default — generalised + phone-aware) ──
    if cmd == "extract":
        run_script("extract_features_v4.py")
    elif cmd == "train":
        run_script("train_v4.py")
    elif cmd == "live":
        run_script("live_test_v5.py")
    elif cmd == "all":
        run_script("extract_features_v4.py")
        run_script("train_v4.py")
        run_script("live_test_v5.py")

    # ── V3 legacy ──
    elif cmd == "extract-v3":
        run_script("extract_features_v3.py")
    elif cmd == "train-v3":
        run_script("train_m_dataset_v3.py")
    elif cmd == "live-v3":
        run_script("live_test_v3.py")
    elif cmd in ("live-rules", "rules", "v4"):
        run_script("live_test_v4.py")
    elif cmd == "all-v3":
        run_script("extract_features_v3.py")
        run_script("train_m_dataset_v3.py")
        run_script("live_test_v4.py")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
