
#SBATCH -p milanq # partition (queue)
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH -t 14-00:00 # time (D-HH:MM)
#SBATCH -o slurm.sqlite_mean.%N.%j.out # STDOUT
#SBATCH -e slurm.sqlite_mean.%N.%j.err # STDERR

echo "Running ocs for sqlite with mean optimization"
srun python src/ocs_ortools.py --system sqlite --optimize_type mean --threads 32
echo "Done"
