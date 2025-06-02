#!/bin/bash
#SBATCH --job-name=notears_bootstrap
#SBATCH --output=notears.out
#SBATCH --error=notears.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=paul  # oder eine andere passende Partition

# Python Module laden
module load Python/3.9

# Venv aktivieren (vorausgesetzt es existiert bereits)
source /home/sc.uni-leipzig.de/og98ohex/notears/notears/venv_notears/bin/activate  # Absolute path!

# Test ob Pakete verfügbar sind
python -c "import numpy, scipy, pandas; print('Packages OK')"

# In Arbeitsverzeichnis wechseln
cd $SLURM_SUBMIT_DIR

# Python Script ausführen
python notears/linear.py

echo "Job completed at $(date)"