import scipy.io
import numpy as np
from scipy.sparse import coo_matrix

data = scipy.io.loadmat(r'..\podaci\PageRankTestData.mat')
# print(data)

G = data['G']   # 'G' graph matrix representation
U = data['U']   # representation of vertices (web pages)
n = 500 ## za G
n_js = G.sum(axis=0)

rows = []
cols = []
values = []
for i, j in zip(*G.nonzero()):
    rows.append(i)
    cols.append(j)
    values.append(1 / n_js[0, j])
A = coo_matrix((values, (rows, cols)), shape=G.shape).tocsc()

alpha = 0.9                 # alpha 
x = np.ones(n) / n          # stohastic vector x_0

k = 100
for i in range(k):          # iterations Mx
    x = (1 - alpha) * A.dot(x) + np.full(n, alpha * np.sum(x) / n)

x = x / np.linalg.norm(x)
pairs = list(zip(x, U))                                                 # pair gotten values with web page representations
sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)    # sort descending by the x_i value

# test print first few
p = 10
for value, label in sorted_pairs:
    print(value, label)
    p -= 1
    if not p: break