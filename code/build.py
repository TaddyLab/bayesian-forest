
import numpy as np
import pandas as pd
import numpy.random as rn
from scipy import sparse

# varnames = pd.read_table('data/E5422/varnames.txt', header=None)
# varnames = varnames.squeeze().tolist()

ydx = pd.read_table('data/E5422/ydx.txt', header=None, sep=" ")
ydx.columns = ['i','j','v']

i = ydx['i'].values-1
j = ydx['j'].values-1
v = ydx['v'].values

ydx = sparse.csr_matrix( (v, (i,j)) )

n = ydx.shape[0]
ind = np.arange(n)
np.random.shuffle(ind)
nb = int(np.ceil(n/128))
indz = [ind[i:i+nb] for i in range(0,n,nb)]

for k in range(128):
	mk = sparse.coo_matrix(ydx[indz[k],:])
	np.savez("data/E5422/users%03d"% k,data=mk.data, row=mk.row, col=mk.col, shape=mk.shape)
	print("users%d: %d rows" % (k,mk.shape[0]))

