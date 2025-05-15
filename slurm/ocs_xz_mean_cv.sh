#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.xz_mean_cv.%N.%j.out # STDOUT
#SBATCH -e slurm.xz_mean_cv.%N.%j.err # STDERR

echo "Running ocs for xz with mean optimization (CV)"
srun python src/ocs_ortools_cv.py --system xz --optimize_type mean --threads 48
echo "Done"
