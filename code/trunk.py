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
	print(i, end=" ")