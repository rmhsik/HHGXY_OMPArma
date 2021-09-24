#!/bin/bash

#SBATCH --job-name="ITP"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --nodelist=nodo07
#SBATCH --ntasks=8
srun time ./ITP && time ./Evolution
