### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
k = sys.argv[1].zfill(3)
F = int(sys.argv[2])
RANDO = int(sys.argv[3])

if F*5 <= int(k) < (F*5+5):
	print("feeling left out")
	sys.exit(0)

import numpy as np
from numpy import random as rn
from scipy import sparse
from sklearn import tree
from sklearn.externals import joblib

ydx = np.load("data/E5422/users%s.npz"% k)
ydx = sparse.csr_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'])

if RANDO==0:
	print("using pre-tree")
	dt = joblib.load('results/%s/fold%d/fit/tree/dtr.pkl'%(J,F)) 
	bvec = dt.tree_.apply(ydx[:,1:].astype(tree._tree.DTYPE))
else:
	print("mapping randomly")
	bvec = rn.random_integers(0,RANDO-1,ydx.shape[0])

for b in set(bvec):
	print(b)
	os.makedirs("results/%s/fold%d/data/%d" % (J,F,b), exist_ok=True)
	mb = sparse.coo_matrix(ydx[bvec==b,:])
	np.savez("results/%s/fold%d/data/%d/map%s" % (J,F,b,k), 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)

 