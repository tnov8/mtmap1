import scipy.io
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

data = scipy.io.loadmat('PageRankTestData.mat')
# print(data)

G = data['G']   # 'G' graph matrix representation
U = data['U']   # representation of vertices (web pages)
n = 500 ## za G
n_js = G.sum(axis=0)

rows = []
cols = []
values = []
for i, j in zip(*G.nonzero()):
    if n_js[0, j] != 0:
        rows.append(i)
        cols.append(j)
        values.append(1 / n_js[0, j])
A = coo_matrix((values, (rows, cols)), shape=G.shape).tocsc()   # matrix A
# print(A.toarray())

alpha = 0.9                 # alpha
S = np.full((n, n), 1/n)    
x = np.ones(n) / n          # stohastic vector x_0

k = 100
for i in range(k):          # iterations Mx
    x = (1 - alpha) * A.dot(x) + alpha * S.dot(x)

x = x / np.linalg.norm(x)   # normalize x (although not necessary for the result)
pairs = list(zip(x, U))     # pair gotten values with web page representations
sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)    # sort descending by the x_i value
for value, label in sorted_pairs:
    print(value, label)