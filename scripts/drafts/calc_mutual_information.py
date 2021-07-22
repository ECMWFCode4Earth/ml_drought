# calc_mutual_information.py
# https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy

import numpy as np

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI


def shan_entropy(c):
    c_normalized = c/float(np.sum(c))
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H


# SKLEARN
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

if __name__ == "__main__":
    A = np.array([ [ 2.0, 140.0, 128.23, -150.5, -5.4 ],
                [ 2.4, 153.11, 130.34, -130.1,-9.5],
                [ 1.2, 156.9, 120.11, -110.45,-1.12 ] ])

    bins = 5 # ?
    n = A.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(A[:,ix], A[:,jx], bins)


