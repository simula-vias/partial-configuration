#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_gosdt_sqlite_q8-q10-q2-q13.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_gosdt_sqlite_q8-q10-q2-q13.%N.%j.err # STDERR

echo "Running WCP for system sqlite with performances ['q8', 'q10', 'q2', 'q13']"
srun uv run python src/rq0_train_single_wcp_1cfg.py --system sqlite --performances q8,q10,q2,q13 --classifier gosdt
echo "Done"
