#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_gosdt_xz_size-time.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_gosdt_xz_size-time.%N.%j.err # STDERR

echo "Running WCP for system xz with performances ['size', 'time']"
srun uv run python src/rq0_train_single_wcp_1cfg.py --system xz --performances size,time --classifier gosdt
echo "Done"
