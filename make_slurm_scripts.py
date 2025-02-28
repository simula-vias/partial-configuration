from pathlib import Path
import json

data_dir = Path("./data")
slurm_dir = Path("./slurm")
threads = 48

systems = json.load(open(data_dir / "metadata.json")).keys()

for system in systems:
    print(system)

    for ot in ["mean", "max", "both"]:
        slurm_file = slurm_dir / f"ocs_{system}_{ot}.sh"
        slurm_file.write_text(f"""#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {threads}
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.{system}_{ot}.%N.%j.out # STDOUT
#SBATCH -e slurm.{system}_{ot}.%N.%j.err # STDERR

echo "Running ocs for {system} with {ot} optimization"
srun python src/ocs_ortools.py --system {system} --optimize_type {ot} --threads {threads}
echo "Done"
""")
