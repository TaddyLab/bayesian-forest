
import numpy as np
from scipy import sparse
import pandas as pd
import json

# varnames = pd.read_table('data/E5422/varnames.txt', header=None)
# varnames = varnames.squeeze().tolist()

ydx = pd.read_table('data/E5422/ydx.txt', header=None, sep=" ")
ydx.columns = ['i','j','v']

i = ydx['i'].values-1
j = ydx['j'].values-1
v = ydx['v'].values

dcsr = sparse.csr_matrix( (v, (i,j)) )

n = dcsr.shape[0]
ind = np.arange(n)
np.random.shuffle(ind)
nb = int(np.ceil(n/256))
indz = [ind[i:i+nb] for i in range(0,n,nb)]

for b in range(256)
	mb = sparse.coo_matrix(dcsr[indz[b],:])
	np.savez("data/E5422/users%d"% b,data=mb.data, row=mb.row, col=mb.col, shape=mb.shape)
	print(b, end=" ")

#x = np.load("data/E5422/users%d.npz"% b)
#x = sparse.csr_matrix( ( x['data'], (x['row'], x['col']) ), shape = x['shape'])
