"""One-off driver: train all six backbones on the 633-image recollected
set and record real test accuracy for each. Does NOT touch the deployed
checkpoint (model/railway_classifier.pt) - every run here writes to its
own _recollected file so the original 70.7%-on-351-images model stays
untouched.

Run from anywhere:
    python statistics/run_all_backbones.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ARCHES = [
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "shufflenet_v2_x0_5",
    "squeezenet1_1",
]

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "model"
SUMMARY_PATH = ROOT / "statistics" / "recollected_accuracy.json"

results = {}

# resume support: skip archs whose history file already exists and has metrics
def existing_result(arch):
    hist = OUT_DIR / f"railway_classifier_{arch}_recollected_history.json"
    if hist.exists():
        try:
            data = json.loads(hist.read_text())
            return data["metrics"]
        except Exception:
            return None
    return None

for arch in ARCHES:
    prior = existing_result(arch)
    if prior is not None:
        print(f"[skip] {arch} already trained -> test_acc {prior['test_accuracy']:.1%}", flush=True)
        results[arch] = prior
        continue

    out_path = OUT_DIR / f"railway_classifier_{arch}_recollected.pt"
    print(f"\n===== training {arch} =====", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "model" / "train.py"),
         "--arch", arch,
         "--output", str(out_path),
         "--seed", "42"],
        cwd=str(ROOT),
    )

    elapsed = time.time() - t0
    print(f"[{arch}] subprocess exit {proc.returncode} after {elapsed/60:.1f} min", flush=True)

    if proc.returncode != 0:
        print(f"[FAIL] {arch} did not complete, stopping driver", flush=True)
        break

    prior = existing_result(arch)
    results[arch] = prior

    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(results, indent=2))

print("\n===== all done =====", flush=True)
print(json.dumps(results, indent=2))
SUMMARY_PATH.parent.mkdir(exist_ok=True)
SUMMARY_PATH.write_text(json.dumps(results, indent=2))
