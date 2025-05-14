#!/bin/sh
#SBATCH -p fpgaq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.odt_gcc.%N.%j.out # STDOUT
#SBATCH -e slurm.odt_gcc.%N.%j.err # STDERR

echo "Running odt for gcc"
srun python src/odt_labelled.py --system gcc --data_dir data --result_dir results --processes 48
echo "Done"
