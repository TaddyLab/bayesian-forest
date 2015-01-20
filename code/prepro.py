### import relevant stuff
import sys
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]
F = int(sys.argv[1])
D = float(sys.argv[2])
os.makedirs("results/%s/fold%d/fit/tree" % (J,F))

import numpy as np
import pandas as pd
from sklearn import tree
from scipy import sparse
import random

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

ydx = []
lo = {F*10 + f for f in range(10)}
kvec = [k for k in range(128) if k not in lo]
ksamp = random.sample(kvec, 32)
for k in ksamp:
	print("adding %03d" % k)
	mk = np.load("data/E5422/users%03d.npz"% k)
	mk = sparse.coo_matrix( ( mk['data'], (mk['row'], mk['col']) ), shape = mk['shape'])
	ydx = sparse.vstack([ydx,mk])

ydx = sparse.csc_matrix(ydx)
ye = ydx[:,0].toarray().squeeze()
Xe = ydx[:,1:]

msl = int(np.ceil(len(ye)/D))
print("%d min samples per leaf" % msl)
dt = tree.DecisionTreeRegressor(min_samples_leaf=msl)
dt.fit(Xe,ye)
tree.export_graphviz(dt,
	out_file="results/%s/fold%d/fit/tree/dtr.dot" % (J,F), 
	feature_names=varnames[1:])
#dot -Tpdf dtr.dot -o dtr.pdf

from sklearn.externals import joblib
joblib.dump(dt, "results/%s/fold%d/fit/tree/dtr.pkl" % (J,F)) 



