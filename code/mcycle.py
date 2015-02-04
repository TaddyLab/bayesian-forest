import sys
# give preference to local update
sys.path.append("~/.local/lib/python3.4/site-packages")

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
import numpy.random as rn

mcycle = pd.read_csv('data/mcycle.csv')
Xm = mcycle['times'].values.reshape(-1,1)
ym = mcycle['accel'].values

def mcycle_plot(mod,title,**kwargs):
    plt.scatter(mcycle['times'],mcycle['accel'],**kwargs)
    plt.xlabel("seconds", fontsize=16)
    plt.ylabel("acceleration", fontsize=16)
    plt.title(title, fontsize=18, y=1.05)
    xgrid = np.arange(0,60,1).reshape(-1,1)
    plt.plot(xgrid, mod.predict(xgrid),color="red",linewidth=2)

# modal cart fit
mcycle_dt = tree.DecisionTreeRegressor(min_samples_leaf=5)
mcycle_dt.fit(Xm, ym)
# single bayesian cart draw
omega = rn.exponential(1,mcycle.shape[0])
mcycle_bt = tree.DecisionTreeRegressor(min_samples_leaf=5)
mcycle_bt.fit(Xm, ym, sample_weight=omega)
# bayesian forest
mcycle_bf = ensemble.RandomForestRegressor(100,min_samples_leaf=5, bootstrap=2)
mcycle_bf.fit(Xm,ym)

fig = plt.figure(figsize=(11,4))


fig.add_subplot(1,3,2)
mcycle_plot(mcycle_bt, " draw", s=omega*20)

fig.add_subplot(1,3,2)
mcycle_plot(mcycle_bt, "Bayesian tree draw", s=omega*20)

fig.add_subplot(1,3,3)
mcycle_plot(mcycle_bf, "Bayesian forest")

fig.subplots_adjust(wspace=.5, bottom=.15)
fig.savefig("graphs/mcycle.pdf", format="pdf", bbox_inches="tight")