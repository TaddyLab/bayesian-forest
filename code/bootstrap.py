### import relevant stuff
import sys
# to give preference to my local overwrites
sys.path.append("~/.local/lib/python3.4/site-packages")
import os
J = os.environ["SLURM_JOB_NAME"]

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy.random as rn
from scipy import sparse

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

ydx = np.load("data/E5422/users0.npz")
ydx = sparse.csc_matrix( ( ydx['data'], (ydx['row'], ydx['col']) ), shape = ydx['shape'] )

for b in range(1,10):
	mb = np.load("data/E5422/users%d.npz"% b)
	mb = sparse.csc_matrix( ( mb['data'], (mb['row'], mb['col']) ), shape = mb['shape'])
	ydx = sparse.vstack(ydx,mb)

ye = ydxmat[:,0].toarray().squeeze()
Xe = ydxmat[:,1:]

dt = tree.DecisionTreeRegressor(min_samples_leaf=ceil(len(ye)/128))
dt.fit(Xe,ye)
tree.export_graphviz(dt,outfile="%s/initree.dot" % J, feature_names=varnames[1:])
#dot -Tpdf initree.dot -o initree.pdf

from sklearn.externals import joblib
joblib.dump(dt, "%s/initree.pkl" % J) 

