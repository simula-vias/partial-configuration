#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_gosdt_x264_etime-cpu.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_gosdt_x264_etime-cpu.%N.%j.err # STDERR

echo "Running WCP for system x264 with performances ['etime', 'cpu']"
srun uv run python src/rq0_train_single_wcp_1cfg.py --system x264 --performances etime,cpu --classifier gosdt
echo "Done"
