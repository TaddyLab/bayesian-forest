import sys
sys.path.append("~/.local/lib/python3.4/site-packages")
sys.path.append("code")
from readpart import readpart

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import tree
from sklearn import ensemble
from sklearn.externals import joblib
trunk = joblib.load("results/bigeg/trunk.pkl")
from glob import glob
import re
import pickle
from copy import copy

branches = {}
blocks = glob("results/bigeg/forest_block*.pkl")
for b in blocks:
	m = re.search('results/bigeg/forest_block_(\d*).pkl', b)
	print((m.group(1), b))
	branches[m.group(1)] = pickle.load(open(b, 'rb'))
	branches[m.group(1)].verbose = 0

rsamps = [pickle.load(open(f, 'rb')) 
	for f in glob("results/bigeg/forest_rsamp*.pkl")]
sampforest = copy(rsamps[0])
sampforest.estimators_ = []
for f in rsamps:
	sampforest.estimators_ += f.estimators_

sampforest.n_estimators = len(sampforest.estimators_)
sampforest.verbose = 0

def missclass(yx):
	n = yx.shape[0]
	y = yx[:,0].toarray().squeeze().astype(int)
	yhat_ebf = np.empty(n, dtype=int)
	bvec = trunk.tree_.apply(yx[:,1:].astype(tree._tree.DTYPE)).astype("str")
	for b in branches.keys():
		yhat_ebf[bvec==b] = branches[b].predict(yx[bvec==b,1:])

	yhat_rsf = sampforest.predict(yx[:,1:])
	mcrsf = np.sum(yhat_rsf != y)
	mcebf = np.sum(yhat_ebf != y)
	return( (n, mcebf, mcrsf) )


parts = np.loadtxt("data/bigeg/shuffle.txt", dtype=int).tolist()
fout = open("results/bigeg/scores.txt", "w")
mc = (0,0,0)

for i in range(500,1500):
	print(i, end=": ")
	yxi = sparse.csr_matrix(readpart(parts[i]))
	mci = missclass(yxi)
	mc = tuple(map(sum,zip(mc,mci)))
	fout.write("%d %d %d\n"%mci)
	print(mc)
	
fout.close()


