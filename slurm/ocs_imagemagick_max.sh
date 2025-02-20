
#SBATCH -p milanq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.imagemagick_max.%N.%j.out # STDOUT
#SBATCH -e slurm.imagemagick_max.%N.%j.err # STDERR

echo "Running ocs for imagemagick with max optimization"
srun python src/ocs_ortools.py --system imagemagick --optimize_type max --threads 32
echo "Done"
