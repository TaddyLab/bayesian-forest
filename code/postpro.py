### import relevant stuff
import sys
import os
J = os.environ["SLURM_JOB_NAME"]
F = int(sys.argv[1])

import numpy as np
import pandas as pd
import glob

yy = np.empty((0,2))
for f in glob.glob("results/%s/fold%d/pred/*.csv" % (J,F)):
	print(f)
	mf = pd.read_csv(f, header=None)
	yy = np.vstack([yy, mf])

rmse = np.sqrt( np.mean( (yy[:,0]-yy[:,1])**2 ) )
with open("results/%s/mse.txt" % J, 'a') as fout:
	fout.write("%.2f\n" % rmse)
