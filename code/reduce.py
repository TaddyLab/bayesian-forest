### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
b = sys.argv[1]
os.makedirs("results/%s/fit/forest%s" % (J,b))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import tree
from sklearn import ensemble
from sklearn.externals import joblib
import glob

maps = glob.glob("results/%s/data/%s/map*.npz" % (J,b))
ydx = []
for m in maps:
	print(m, end = ": ")
	mk = np.load(m)
	mk = sparse.coo_matrix( ( mk['data'], (mk['row'], mk['col']) ), 
		shape = mk['shape'])
	print(str(mk.shape[0]) + " rows")
	ydx = sparse.vstack([ydx,mk])

ydx = sparse.csc_matrix(ydx)
ye = ydx[:,0].toarray().squeeze()
Xe = ydx[:,1:]

bf = ensemble.RandomForestRegressor(200,
		min_samples_leaf=10,n_jobs=-1,verbose=5,bootstrap=2)
bf.fit(Xe,ye)

joblib.dump(bf, "results/%s/fit/forest%s/bfr.pkl" % (J,b)) 

# # note that the backend="threading" is 'hardcoded into the code 
# # because tree is internally releasing the Python GIL making 
# # threading always more efficient than multiprocessing".  
# # In experiments, everything does seem to take the same time.
# from sklearn.externals.joblib import Parallel, delayed
# # 
# #from time import sleep
# #r = Parallel(n_jobs=5, verbose=5, backend="threading")(delayed(sleep)(1) for _ in range(10))
# ## doesn't work
# def fbf():
# 	bf = ensemble.RandomForestRegressor(125,
# 		min_samples_leaf=10,bootstrap=2,n_jobs=10,verbose=5)
# 	bf.fit(Xe,ye)
# 	return bf

# BF = Parallel(n_jobs=16, verbose=5)(delayed(fbf)() for _ in range(16))



















