### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
k = sys.argv[1].zfill(3)
F = int(sys.argv[2])
if F*10 <= int(k) < (F*10+10):
	print("feeling left out")
	sys.exit(0)

import numpy as np
from scipy import sparse
from sklearn import tree
from sklearn.externals import joblib

dt = joblib.load('results/%s/fold%d/fit/tree/dtr.pkl'%(J,F)) 

ydx = np.load("data/E5422/users%s.npz"% k)
ydx = sparse.csr_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'])

bvec = [str(b) for b in dt.tree_.apply(ydx[:,1:].astype(tree._tree.DTYPE))]
for b in set(bvec):
	print(b)
	os.makedirs("results/%s/fold%d/data/%s" % (J,F,b), exist_ok=True)
	mb = sparse.coo_matrix(ydx[bvec==b,:])
	np.savez("results/%s/fold%d/data/%s/map%s" % (J,F,b,k), 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)

