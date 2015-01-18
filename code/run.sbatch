#!/bin/bash
## run with `sbatch -N[#machines] -J[jobname] code/run.sbatch'
## `jobname' determines the result directory name, and will overwrite any existing
#SBATCH --time=36:0:0
#SBATCH --mem-per-cpu=2000
#SBATCH --output=%j.err
#SBATCH --constraint=ib
#SBATCH --exclusive

module load parallel

OUT=results/$SLURM_JOB_NAME
LOG=$OUT/log/an_sbatch.log

mkdir -p $OUT
rm -rf $OUT
mkdir -p $OUT/data
mkdir -p $OUT/log

echo `date` > $LOG
echo "jobid $SLURM_JOBID running $SCRIPT" >> $LOG
echo "$SLURM_NNODES nodes" >> $LOG

export para="parallel -j $SLURM_NNODES --joblog $OUT/log/para.log"
export srun="srun --exclusive -N1 -c16 --ntasks=1"  

python code/prepro.py > $OUT/log/prepro.log

echo "mapping @ `date`" >> $LOG
$para $srun "python code/map.py {1} > $OUT/log/map{1}.log" ::: {0..127}

echo "reducing @ `date`" >> $LOG
blocks=$(ls results/$SLURM_JOB_NAME/data/ | cut -d'/' -f -1)
$para $srun "python code/reduce.py {1} > $OUT/log/reduce{1}.log" ::: $blocks

echo "done @" `date` >> $LOG


