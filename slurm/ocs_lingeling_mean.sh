#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.lingeling_mean.%N.%j.out # STDOUT
#SBATCH -e slurm.lingeling_mean.%N.%j.err # STDERR

echo "Running ocs for lingeling with mean optimization"
srun python src/ocs_ortools.py --system lingeling --optimize_type mean --threads 48
echo "Done"
