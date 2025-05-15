#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.x264_mean_cv.%N.%j.out # STDOUT
#SBATCH -e slurm.x264_mean_cv.%N.%j.err # STDERR

echo "Running ocs for x264 with mean optimization (CV)"
srun python src/ocs_ortools_cv.py --system x264 --optimize_type mean --threads 48
echo "Done"
