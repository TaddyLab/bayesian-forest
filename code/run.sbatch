#!/bin/bash
## run with `sbatch -N[#machines] -J[jobname] code/run.sbatch'
## `jobname' determines the result directory name, and will overwrite any existing
#SBATCH --time=36:0:0
#SBATCH --mem-per-cpu=2000
#SBATCH --output=%j.err
#SBATCH --constraint=ib
#SBATCH --exclusive

PRETREE=1
BOOTSTRAP=2
NTREE=16

module load parallel
OUT=results/$SLURM_JOB_NAME
LOG=$OUT/an_sbatch.log

mkdir -p $OUT
rm -rf $OUT/*

mkdir -p $OUT/code
cp code/*.py $OUT/code/.

echo `date` > $LOG
echo "jobid $SLURM_JOBID running $SCRIPT" >> $LOG
echo "$SLURM_NNODES nodes" >> $LOG
echo "pretree=$PRETREE for mapping" >> $LOG
echo "bootstrap=$BOOTSTRAP in forest training" >> $LOG
echo "$NTREE trees per mini-forest" >> $LOG

export srun="srun --exclusive -N1 -c16 --ntasks=1"  

for F in {0..9}; do
	export para="parallel -j $SLURM_NNODES --joblog $OUT/fold$F/log/para.log"

	echo "*****fold=$F" >> $LOG
	mkdir -p $OUT/fold$F/log
	mkdir -p $OUT/fold$F/data
	mkdir -p $OUT/fold$F/fit
	mkdir -p $OUT/fold$F/pred

 	if [ "$PRETREE" -eq 1 ]; then 
 		echo "pre-tree @ `date`" >> $LOG
 		python code/prepro.py $F > $OUT/fold$F/log/prepro.log; fi

 	echo "mapping @ `date`" >> $LOG
 	$para $srun "python code/map.py {1} $F $PRETREE > $OUT/fold$F/log/map{1}.log" ::: {0..127}

 	echo "reducing @ `date`" >> $LOG
 	blocks=$(ls results/$SLURM_JOB_NAME/fold$F/data/ | cut -d'/' -f -1)
 	$para $srun "python code/reduce.py {1} $F $BOOTSTRAP $NTREE > $OUT/fold$F/log/reduce{1}.log" ::: $blocks

 	echo "predicting @ `date`" >> $LOG
 	$para $srun "python code/pred.py {1} $F > $OUT/fold$F/log/pred{1}.log" ::: {0..127}
	python code/postpro.py $F > $OUT/fold$F/log/postpro.log
done

echo "done @" `date` >> $LOG


