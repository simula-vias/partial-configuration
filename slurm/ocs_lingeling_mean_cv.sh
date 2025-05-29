#!/bin/sh
#SBATCH -p rome16q # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 21
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.lingeling_mean_cv.%N.%j.out # STDOUT
#SBATCH -e slurm.lingeling_mean_cv.%N.%j.err # STDERR

echo "Running ocs for lingeling with mean optimization (CV)"
srun python src/ocs_ortools_cv.py --system lingeling --optimize_type mean --threads 32
echo "Done"
