#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.x264_both.%N.%j.out # STDOUT
#SBATCH -e slurm.x264_both.%N.%j.err # STDERR

echo "Running ocs for x264 with both optimization"
srun python src/ocs_ortools.py --system x264 --optimize_type both --threads 48
echo "Done"
