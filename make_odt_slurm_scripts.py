from pathlib import Path
import json

data_dir = Path("./data")
slurm_dir = Path("./slurm")
result_dir = Path("./results")
threads = 48

systems = json.load(open(data_dir / "metadata.json")).keys()

for system in systems:
    print(system)
    slurm_file = slurm_dir / f"odt_{system}.sh"
    slurm_file.write_text(f"""#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {threads}
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.odt_{system}.%N.%j.out # STDOUT
#SBATCH -e slurm.odt_{system}.%N.%j.err # STDERR

echo "Running odt for {system}"
srun python src/odt_labelled.py --system {system} --data_dir {data_dir} --result_dir {result_dir} --processes {threads}
echo "Done"
""")
