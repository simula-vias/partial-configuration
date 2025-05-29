#!/bin/sh
#SBATCH -p rome16q # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.lingeling_both_cv.%N.%j.out # STDOUT
#SBATCH -e slurm.lingeling_both_cv.%N.%j.err # STDERR

echo "Running ocs for lingeling with both optimization (CV)"
srun python src/ocs_ortools_cv.py --system lingeling --optimize_type both --threads 32
echo "Done"
