
import itertools
import json
from pathlib import Path

data_dir = Path("./data")
systems = json.load(open(data_dir / "metadata.json")).keys()

for s in systems:
    # if s != "nodejs":
    #     continue
    # if s in ("lingeling", "nodejs"):
    #     continue

    # Load all performances for the system
    # This is a bit inefficient as we load the whole data just for the performances
    # A better way would be to have this information in the metadata
    from common import load_data
    (
        _,
        _,
        _,
        all_performances_initial,
        _,
        _,
    ) = load_data(system=s, data_dir=data_dir)

    all_perf_list = [[ap] for ap in all_performances_initial]

    for num_p in range(2, len(all_performances_initial) + 1):
        if s == "sqlite":
            # sqlite has too many performance measures
            all_perf_list.append(all_performances_initial[:num_p])
        else:
            all_perf_list.extend(
                list(map(list, itertools.combinations(all_performances_initial, num_p)))
            )
    
    for performances in all_perf_list:
        for classifier in ["dt", "rf", "gosdt", "dl85"]:
            cmd = [
                "python",
                "src/rq0_train_single_wcp_1cfg.py",
                "--system",
                s,
                "--performances",
                ",".join(performances),
                "--classifier",
                classifier,
            ]
            print(f"{' '.join(cmd)}")

            script = f"""#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_{classifier}_{s}_{'-'.join(performances)}.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_{classifier}_{s}_{'-'.join(performances)}.%N.%j.err # STDERR

echo "Running WCP for system {s} with performances {performances}"
srun uv run {' '.join(cmd)}
echo "Done"
"""
            open(f"./slurm/wcp_{classifier}_{s}_{'-'.join(performances)}.sh", "w").write(script)
