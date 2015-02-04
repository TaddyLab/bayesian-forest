## run with, e.g.
#samps=$(ls data/bigeg/rsamp*.npz | awk -F '[/.]' '{print $3}')
#for s in $samps; do echo $s; python code/forest.py $s; done

import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
samp = sys.argv[1]

MSL=100
NJ=20
NTREE=100

import numpy as np
from scipy import sparse
from sklearn import ensemble
from sklearn.externals import joblib
import pickle

yx = np.load("data/bigeg/%s.npz"% samp)
yx = sparse.csc_matrix( 
	( yx['data'], (yx['row'], yx['col']) ), 
	shape = yx['shape'])

y = yx[:,0].toarray().squeeze().astype(int)

forest = ensemble.RandomForestClassifier(
		NTREE, min_samples_leaf=MSL,n_jobs=NJ,verbose=2)
forest.fit(yx[:,1:],y)

pickle.dump(forest, open("results/bigeg/forest_%s.pkl" % samp, 'wb'))















