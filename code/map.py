### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
k = sys.argv[1].zfill(3)

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import tree
from sklearn import ensemble
from sklearn.externals import joblib

dt = joblib.load('results/%s/fit/tree/dtr.pkl'%J) 

ydx = np.load("data/E5422/users%s.npz"% k)
ydx = sparse.csr_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'])

bvec = dt.tree_.apply(ydx[:,1:].astype(tree._tree.DTYPE))
for b in set(bvec):
	print(b)
	os.makedirs("results/%s/data/%d" % (J,b), exist_ok=True)
	mb = sparse.coo_matrix(ydx[bvec==b,:])
	np.savez("results/%s/data/%d/map%s" % (J,b,k), 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)

