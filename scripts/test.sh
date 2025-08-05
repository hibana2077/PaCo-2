#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=16GB           
#PBS -l walltime=00:02:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/PaCo-2/.venv/bin/activate
python3 test_optimization.py --batch-size 4 --iterations 2 >> out_test.txt

