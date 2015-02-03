import sys
import numpy as np
import pandas as pd
from sklearn import tree
from scipy import sparse
import random
import itertools

sys.path.append('code')
from readpart import readpart

parts = np.arange(1500)
random.shuffle(parts)
lo = parts[1e3:]
with open("data/bigeg/validate.txt", 'w') as fout:
	for i in lo:
		fout.write("%d\n"%i)

yx = []
for i in range(1000):
	yx = sparse.vstack([yx, readpart(parts[i])])
	print(i)

yx = sparse.csc_matrix(yx)
y = yx[:,0].toarray().squeeze()

## fit trunk
n = yx.shape[0]
msl = int(np.ceil(n/100.0))
print("%d min samples per leaf" % msl)
trunk = tree.DecisionTreeClassifier(min_samples_leaf=msl)
trunk.fit(y, yx[:,1:])

## output
varnames = pd.read_table('data/bigeg/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()
tree.export_graphviz(trunk,
	out_file="results/bigeg/trunk.dot", 
	feature_names=varnames[1:])
from sklearn.externals import joblib
joblib.dump(dt, "results/bigeg/trunk.pkl") 

## send everyone 
yx = sparse.csr_matrix(yx)
bvec = dt.tree_.apply(yx[:,1:].astype(tree._tree.DTYPE))
for b in set(bvec):
	print(b)
	mb = sparse.coo_matrix(yx[bvec==b,:])
	np.savez("block_%d" % b, 
		data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)

## random allocations for comparison





















