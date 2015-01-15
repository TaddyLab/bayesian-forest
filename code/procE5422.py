
import numpy as np
import scipy as sp
import pandas as pd
import json

varnames = pd.read_table('data/E5422/varnames.txt', header=None)
varnames = varnames.squeeze().tolist()

users = {}
r = 0
ydx = open('data/E5422/ydx.txt','r')

for line in ydx:
	i,j,x = line.split() 
	if i in users:
		users[i][varnames[int(j)-1]] = float(x)
	else:
		users[i] = {varnames[int(j)-1]:float(x)}
	r += 1
	if r % 1e6 == 0:
		print("%.2e" % r)

users = [json.dumps(users[i]) for i in users]
with open("data/E5422/users.json", 'w') as fout:
	for r in users:
		fout.write(r+'\n')

# mkdir data/E5422/users
# split -l $(( $( wc -l < data/E5422/users.json ) / 256 + 1 )) -a 3 -d  data/E5422/users.json  data/E5422/users/part

