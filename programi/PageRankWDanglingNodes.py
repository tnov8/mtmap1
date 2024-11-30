import time
import numpy as np
from scipy.sparse import coo_matrix

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

# read data
with open('..\\podaci\\enron.txt', "r") as file:
    lines = file.readlines()
n = int(lines[0].strip())
m = int(lines[1].strip())
rows = []
cols = []
values = []
for i in range(2, 2 + m):
    x, y = map(int, lines[i].strip().split())
    rows.append(y - 1)
    cols.append(x - 1)
    values.append(1)

# find non-dangling nodes
perm = list(set(cols))
k = len(perm)
print("n - nodes:", n)
print("m - non-zero entries: ", m, " (", round(m / (n * n), 6) * 100, "%)", sep = '')
print("k - non-dangling nodes:", k)
print("Dangling nodes: ", n - k, " (", (100 * (n - k)) // n, "%)", sep = '')
print()

# fill perm with remaining nodes
remaining = [1] * n
for i in perm:
    remaining[i] = 0
for i in range(n):
    if remaining[i]: perm.append(i)

# rearrange nodes
rows2 = []
cols2 = []
inverse_perm = inv(perm)
for i, j in zip(rows, cols):
    rows2.append(inverse_perm[i])
    cols2.append(inverse_perm[j])
A = coo_matrix((values, (rows2, cols2)), shape=(n,n)).tocsc()


# form H with normalized values from A
n_js = A.sum(axis=0)
rows = []
cols = []
values = []
for i, j in zip(*A.nonzero()):
    rows.append(i)
    cols.append(j)
    values.append(1 / n_js[0, j])
H = coo_matrix((values, (rows, cols)), shape=(n,n)).tocsc()


# POWER METHOD WITH LUMPING

alpha = 0.85
v = np.ones(n) / n
v1 = v[:k]
w = np.ones(n) / n
w1 = w[:k]
sigma = np.ones(k + 1) / (k + 1)
sigma1 = sigma[:k]
sigma2 = sigma[k]
H_11 = H[:k, :k]
H_12 = H[k:, :k]
pi = np.ones(n)

iterations = 500
print(iterations, "iterations")

start_time = time.time()

for i in range(iterations):
    sigma1 = alpha * H_11.dot(sigma1) + (1 - alpha) * v1 + alpha * sigma2 * w1
    sigma2 = 1 - np.sum(sigma1)
pi[:k] = sigma1
pi[k:] = alpha * H_12.dot(sigma1) + (1 - alpha) * v[k:] + alpha * sigma2 * w[k:]

end_time = time.time()
execution_time = end_time - start_time
print(f"Power method with lumping: {execution_time:.6f}s")


# REGULAR POWER METHOD

x = np.ones(n) / n

start_time = time.time()

for i in range(iterations):
    x = (1 - alpha) * H.dot(x) + np.full(n, alpha * np.sum(x) / n)

end_time = time.time()
execution_time = end_time - start_time
print(f"Regular power method: {execution_time:.6f}s")
print()


# inverse permutation to get original nodes
sorted_pi = np.argsort(-pi)
# print(sorted_pi[:10])
PageRank = []
for z in sorted_pi: PageRank.append(perm[z])
print("Power method with lumping PageRank (first 10):", PageRank[:10])

sorted_x = np.argsort(-x)
# print(sorted_x[:10])
PageRank_ = []
for z in sorted_x: PageRank_.append(perm[z])
print("Regular power method PageRank (first 10):     ", PageRank_[:10])

