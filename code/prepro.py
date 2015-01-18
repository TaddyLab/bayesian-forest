### import relevant stuff
import sys
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
os.makedirs("results/%s/fit/tree" % J)

import numpy as np
import pandas as pd
from sklearn import tree
from scipy import sparse

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

ydx = np.load("data/E5422/users000.npz")
ydx = sparse.coo_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'] )

Kprep = 12
for k in range(1,Kprep):
	mk = np.load("data/E5422/users%03d.npz"% k)
	mk = sparse.coo_matrix( ( mk['data'], (mk['row'], mk['col']) ), shape = mk['shape'])
	ydx = sparse.vstack([ydx,mk])
	print(k)

ydx = sparse.csc_matrix(ydx)
ye = ydx[:,0].toarray().squeeze()
Xe = ydx[:,1:]

msl = int(np.ceil(len(ye)/128))
dt = tree.DecisionTreeRegressor(min_samples_leaf=msl)
dt.fit(Xe,ye)
tree.export_graphviz(dt,out_file="results/%s/fit/tree/dtr.dot" % J, feature_names=varnames[1:])
#dot -Tpdf dtr.dot -o dtr.pdf

from sklearn.externals import joblib
joblib.dump(dt, "results/%s/fit/tree/dtr.pkl" % J) 



