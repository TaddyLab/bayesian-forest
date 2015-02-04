## run with, e.g.
#samps=$(ls data/bigeg/rsamp*.npz | awk -F '[/.]' '{print $3}')
#for s in $samps; do echo $s; python code/forest.py $s; done

import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
samp = sys.argv[1]

MSL=1000
NJ=10
NTREE=100

import numpy as np
from scipy import sparse
from sklearn import ensemble
from sklearn.externals import joblib
import pickle

ydx = np.load("data/bigeg/%s.npz"% samp)
ydx = sparse.csc_matrix( 
	( ydx['data'], (ydx['row'], ydx['col']) ), 
	shape = ydx['shape'])

y = ydx[:,0].toarray().squeeze().astype(int)

forest = ensemble.RandomForestClassifier(
		NTREE, min_samples_leaf=MSL,n_jobs=NJ,verbose=2)
forest.fit(ydx[:,1:],y)

pickle.dump(forest, open("results/bigeg/forest_%s.pkl" % samp, 'wb'))















