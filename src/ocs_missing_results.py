from pathlib import Path
import json

data_dir = Path("./data")
results_dir = Path("./results")

systems = json.load(open(data_dir / "metadata.json")).keys()

for system in systems:
    for ot in ["both", "mean"]:
        full_file = results_dir / f"ocs_{system}_{ot}.json"

        if not full_file.exists():
            print(f"Result file {full_file} missing.")

        cv_file = results_dir / f"ocs_{system}_{ot}_cv.json"

        if not cv_file.exists():
            print(f"Result file {cv_file} missing.")

