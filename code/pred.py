### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
k = sys.argv[1].zfill(3)
F = int(sys.argv[2])
if (F*10 > int(k)) or (int(k) >= (F*10+10)):
	print("i'm IN sample... take a hike.")
	sys.exit(0)

import numpy as np
from scipy import sparse
from sklearn import tree
from sklearn import ensemble
from sklearn.externals import joblib
import pickle

dt = joblib.load('results/%s/fold%d/fit/tree/dtr.pkl'%(J,F)) 
ydx = np.load("data/E5422/users%s.npz"% k)
ydx = sparse.csr_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'])
bvec = dt.tree_.apply(ydx[:,1:].astype(tree._tree.DTYPE))

for b in set(bvec):
	print(b)
	#bfr = joblib.load("results/%s/fold%d/fit/forest%d/bfr.pkl" % (J,F,b)) 
	bfr = pickle.load( open("results/%s/fold%d/fit/forest%d/bfr.pkl" % (J,F,b), 'rb') ) 
	isb = bvec==b
	pb = bfr.predict(ydx[isb,1:])
	yb = ydx[isb,0].toarray()
	with open("results/%s/fold%d/pred/chunk%sblock%d.csv" % (J,F,k,b), 'w') as fout:
		for i in range(len(pb)):
			fout.write("%f,%f\n" % (yb[i],pb[i]))

