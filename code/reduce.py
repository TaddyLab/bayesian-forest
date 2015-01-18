### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
b = sys.argv[1]

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import tree
from sklearn import ensemble
from sklearn.externals import joblib
import glob

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

maps = glob.glob("results/%s/data/block%s/map*.npz" % (J,b))
ydx = []
for m in maps:
	print(m, end = ": ")
	mk = np.load(m)
	mk = sparse.coo_matrix( ( mk['data'], (mk['row'], mk['col']) ), shape = mk['shape'])
	print(str(mk.shape[0]) + " rows")
	ydx = sparse.vstack([ydx,mk])

