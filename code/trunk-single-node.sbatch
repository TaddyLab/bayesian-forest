#!/bin/bash

#SBATCH --job-name=spark
#SBATCH --exclusive
#SBATCH --nodes=1

module load spark

export MASTER="local[*]"

spark-submit --master $MASTER sparktrunk.py

