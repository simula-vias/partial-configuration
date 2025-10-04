#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_rf_lingeling_reductions.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_rf_lingeling_reductions.%N.%j.err # STDERR

echo "Running WCP for system lingeling with performances ['reductions']"
srun uv run python src/rq0_train_single_wcp_1cfg.py --system lingeling --performances reductions --classifier rf
echo "Done"
