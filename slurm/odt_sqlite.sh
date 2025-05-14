#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.odt_sqlite.%N.%j.out # STDOUT
#SBATCH -e slurm.odt_sqlite.%N.%j.err # STDERR

echo "Running odt for sqlite"
srun python src/odt_labelled.py --system sqlite --data_dir data --result_dir results --processes 48
echo "Done"
