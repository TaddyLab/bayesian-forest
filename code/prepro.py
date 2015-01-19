### import relevant stuff
import sys
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
F = int(sys.argv[1])
os.makedirs("results/%s/fold%d/fit/tree" % (J,F))

import numpy as np
import pandas as pd
from sklearn import tree
from scipy import sparse

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

ydx = []
K = 28
kstart = F*10+10
for k in range(kstart,kstart+K):
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
tree.export_graphviz(dt,
	out_file="results/%s/fold%d/fit/tree/dtr.dot" % (J,F), 
	feature_names=varnames[1:])
#dot -Tpdf dtr.dot -o dtr.pdf

from sklearn.externals import joblib
joblib.dump(dt, "results/%s/fold%d/fit/tree/dtr.pkl" % (J,F)) 



