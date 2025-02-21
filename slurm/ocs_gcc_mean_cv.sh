#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.gcc_mean.%N.%j.out # STDOUT
#SBATCH -e slurm.gcc_mean.%N.%j.err # STDERR

echo "Running ocs for gcc with mean optimization"
srun python src/ocs_ortools_cv.py --system gcc --optimize_type mean --threads 48 -k 4
echo "Done"
