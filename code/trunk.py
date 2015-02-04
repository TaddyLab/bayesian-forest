import sys
sys.path.append('code')
from readpart import readpart

sys.path.append("~/.local/lib/python3.4/site-packages")
from sklearn import tree

from sklearn.externals import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import random


parts = np.arange(1500)
random.shuffle(parts)
with open("data/bigeg/shuffle.txt", "w") as fout:
	for p in parts:
		fout.write("%d\n"%p)

buffmax = int(1e10)
yx_data = np.empty(buffmax)
yx_row = np.empty(buffmax, dtype=int)
yx_col = np.empty(buffmax, dtype=int)
n = 0
l = 0
for i in range(500):
	yxi = readpart(parts[i])
	li = len(yxi.row)
	yx_data[l:(l+li)] = yxi.data
	yx_row[l:(l+li)] = yxi.row + n
	yx_col[l:(l+li)] = yxi.col
	n += yxi.shape[0]
	l += li
	print((i,n,l))

yx_data = yx_data[:l].astype(np.float32)
yx_row = yx_row[:l]
yx_col = yx_col[:l]
yx = sparse.csc_matrix((yx_data, (yx_row,yx_col)))
yx_data = None
yx_row = None
yx_col = None

joblib.dump(yx, "data/bigeg/yx.pkl")
y = yx[:,0].toarray().squeeze().astype(int)

## fit trunk
msl = 250000
trunk = tree.DecisionTreeRegressor(min_samples_leaf=msl)
trunk.fit(yx[:,1:],y)

joblib.dump(trunk, "results/bigeg/trunk.pkl") 

## output
varnames = pd.read_table('data/bigeg/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()
tree.export_graphviz(trunk,
	out_file="results/bigeg/trunk.dot", 
	feature_names=varnames[1:])

## send everyone 
yx = sparse.csr_matrix(yx)
bvec = trunk.tree_.apply(yx[:,1:].astype(trunk._tree.DTYPE))
for b in set(bvec):
	print(b)
	mb = sparse.coo_matrix(yx[bvec==b,:])
	np.savez("block_%d" % b, 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)

## random allocations for comparison
bvec = rn.random_integers(0,len(set(bvec))-1,yx.shape[0])
for b in set(bvec):
	print(b)
	mb = sparse.coo_matrix(yx[bvec==b,:])
	np.savez("rsamp_%d" % b, 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)



















