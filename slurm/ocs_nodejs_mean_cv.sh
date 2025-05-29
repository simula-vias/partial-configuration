#!/bin/sh
#SBATCH -p rome16q # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.nodejs_mean_cv.%N.%j.out # STDOUT
#SBATCH -e slurm.nodejs_mean_cv.%N.%j.err # STDERR

echo "Running ocs for nodejs with mean optimization (CV)"
srun python src/ocs_ortools_cv.py --system nodejs --optimize_type mean --threads 32
echo "Done"
