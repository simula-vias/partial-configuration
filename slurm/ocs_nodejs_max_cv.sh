#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.nodejs_max.%N.%j.out # STDOUT
#SBATCH -e slurm.nodejs_max.%N.%j.err # STDERR

echo "Running ocs for nodejs with max optimization"
srun python src/ocs_ortools_cv.py --system nodejs --optimize_type max --threads 48 -k 4
echo "Done"
