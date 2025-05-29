#!/bin/bash
#SBATCH --job-name=notears_bootstrap
#SBATCH --output=notears.out
#SBATCH --error=notears.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Python Module laden (prüfe verfügbare Module mit: module avail python)
module load Python/3.9

# Venv aktivieren oder neu erstellen falls Probleme
if [ ! -f "venv_notears/bin/activate" ]; then
    echo "Creating new virtual environment..."
    python -m venv venv_notears
    source venv_notears/bin/activate
    pip install numpy scipy pandas python-igraph
else
    echo "Using existing virtual environment..."
    source venv_notears/bin/activate
fi

# Test ob Pakete verfügbar sind
python -c "import numpy, scipy, pandas; print('Packages OK')"

# In dein Arbeitsverzeichnis wechseln
cd $SLURM_SUBMIT_DIR

# Python Script ausführen
python notears/linear.py

echo "Job completed at $(date)"