#!/bin/sh
#SBATCH -p slowq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.wcp_dl85_sqlite_q8-q10-q2-q13-q5-q7-q1-q3-q9-q14-q6-q12-q4-q15-q11.%N.%j.out # STDOUT
#SBATCH -e slurm.wcp_dl85_sqlite_q8-q10-q2-q13-q5-q7-q1-q3-q9-q14-q6-q12-q4-q15-q11.%N.%j.err # STDERR

echo "Running WCP for system sqlite with performances ['q8', 'q10', 'q2', 'q13', 'q5', 'q7', 'q1', 'q3', 'q9', 'q14', 'q6', 'q12', 'q4', 'q15', 'q11']"
srun uv run python src/rq0_train_single_wcp_1cfg.py --system sqlite --performances q8,q10,q2,q13,q5,q7,q1,q3,q9,q14,q6,q12,q4,q15,q11 --classifier dl85
echo "Done"
