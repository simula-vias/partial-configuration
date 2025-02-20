
#SBATCH -p milanq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.poppler_mean.%N.%j.out # STDOUT
#SBATCH -e slurm.poppler_mean.%N.%j.err # STDERR

echo "Running ocs for poppler with mean optimization"
srun python src/ocs_ortools.py --system poppler --optimize_type mean --threads 32
echo "Done"
