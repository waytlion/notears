#!/bin/bash
#SBATCH --job-name=notears_bootstrap
#SBATCH --output=notears_%j.out
#SBATCH --error=notears_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Python Module laden (prüfe verfügbare Module mit: module avail python)
module load Python/3.9

# Virtual Environment aktivieren
source venv_notears/bin/activate

# In dein Arbeitsverzeichnis wechseln
cd $SLURM_SUBMIT_DIR

# Python Script ausführen
python linear.py

echo "Job completed at $(date)"