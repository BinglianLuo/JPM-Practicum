import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# %%
# dt1 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/PreprocessTest/T1.xlsx')
dt1 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA1.xlsx')
dt1['Mid'] = (dt1['Bid'] + dt1['Ask']) / 2
S0 = 918.4
K = dt1['Strike']
dt1 = dt1 / S0
S0 = 1

# dt2 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/PreprocessTest/T2.xlsx')
dt2 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA2.xlsx')
dt2['Mid'] = (dt2['Bid'] + dt2['Ask']) / 2
S0 = 918.4
K = dt2['Strike']
dt2 = dt2 / S0
S0 = 1

# dt3 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/PreprocessTest/T3.xlsx')
dt3 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA3.xlsx')
dt3['Mid'] = (dt3['Bid'] + dt3['Ask']) / 2
S0 = 918.4
K = dt3['Strike']
dt3 = dt3 / S0
S0 = 1

# %%
n1 = len(dt1)
n2 = len(dt2)
n3 = len(dt3)
N = n1 + n2 + n3
n = [n1, n2, n3]
m = len(n)

k = [list(dt1['Strike']), list(dt2['Strike']), list(dt3['Strike'])]
c_ask = list(dt1['Ask']) + list(dt2['Ask']) + list(dt3['Ask'])
c_bid = list(dt1['Bid']) + list(dt2['Bid']) + list(dt3['Bid'])
c_mid = list(dt1['Mid']) + list(dt2['Mid']) + list(dt3['Mid'])

delta_bid = [(b - a) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
delta0 = min(1 / N, min(delta_ask), min(delta_bid))

del n1, n2, n3
# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = np.array([[0,0,0]])
# A = np.r_[A,b]

# %%
A = np.zeros(shape=(0, 2 * N))
b = []

# C1 Outright: m
for i in range(0, m):
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:(i + 1)]) - 1] = 1
    A = np.r_[A, row]
    b.append(0)

# %%
# C2 Vertical spread: N + m
for i in range(0, m):
    # VS_0 >= 0, -c1 >= -1
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:i])] = -1
    A = np.r_[A, row]
    b.append(-1)

    # VS_0 <= 1, c1 >= 1 - k1
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:i])] = 1
    A = np.r_[A, row]
    b.append(1 - k[i][0])

    # VS_j >= 0, c1 - c2 >= 0
    for j in range(0, n[i] - 1):
        row = np.zeros(shape=(1, 2 * N))
        row[0][sum(n[0:i]) + j] = 1
        row[0][sum(n[0:i]) + j + 1] = -1
        A = np.r_[A, row]
        b.append(0)

# %% C3 Vertical butterfly: N - m
for i in range(0, m):
    for j in range(0, n[i] - 1):

        # -k2*c1 + k1*c2 >= k1 - k2
        if j == 0:
            row = np.zeros(shape=(1, 2 * N))
            row[0][sum(n[0:i])] = -k[i][1]
            row[0][sum(n[0:i]) + 1] = k[i][0]
            A = np.r_[A, row]
            b.append(k[i][0] - k[i][1])

        # (k3-k2)c1 + (k1-k3)c2 + (k2-k1)c3 >= 0
        else:
            row = np.zeros(shape=(1, 2 * N))
            row[0][sum(n[0:i]) + j - 1] = -k[i][j] + k[i][j + 1]
            row[0][sum(n[0:i]) + j] = k[i][j - 1] - k[i][j + 1]
            row[0][sum(n[0:i]) + j + 1] = k[i][j] - k[i][j - 1]
            A = np.r_[A, row]
            b.append(0)

# %% C4 Calendar spread
compute_count = 0
A_count = 0
for i1 in range(0, m - 1):
    for j1 in range(0, n[i1]):
        for i2 in range(i1 + 1, m):
            for j2 in range(0, n[i2]):
                compute_count += 1
                if k[i1][j1] == k[i2][j2]:
                    row = np.zeros(shape=(1, 2 * N))
                    row[0][sum(n[0:i1]) + j1] = -1
                    row[0][sum(n[0:i2]) + j2] = 1
                    A = np.r_[A, row]
                    A_count += 1
                    b.append(0)
                    # print(i1, j1, k[i1][j1])
                    # print(i2, j2, k[i2][j2])
                    break

print("Compute_count = ", compute_count)
print("Constraint_count = ", A_count)

del i, j, i1, i2, j1, j2

# %%
c = c_mid + [0] * N
b = list(-matrix(b) + matrix(A) * matrix(c))
A = -A
obj = [0] * N + [1] * N
#%%
for j in range(0, N):
    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = -1
    row[0][N + j] = -1
    A = np.r_[A, row]
    b.append(delta_bid[j] - delta0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = 1
    row[0][N + j] = -1
    A = np.r_[A, row]
    b.append(delta_ask[j] - delta0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = -delta0 / delta_bid[j]
    row[0][N + j] = -1
    A = np.r_[A, row]
    b.append(0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = delta0 / delta_ask[j]
    row[0][N + j] = -1
    A = np.r_[A, row]
    b.append(0)

# %%
obj = matrix(obj, (len(obj), 1), 'd')
b = matrix(b, (len(b), 1), 'd')

sol = solvers.lp(matrix(obj), matrix(A), matrix(b))

# %%
# e = [round(num, 4) for num in list(sol['x'])]
e = list(sol['x'])
print(e)
c = [(a + b) for a, b in zip(c, e)]


# %% Error
plt.scatter([i for i in range(n[0])], e[0:n[0]], c='b', marker='o', s=10, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='-', alpha = 0.5)
plt.title("Call Price Perturbation Value")
plt.ylabel('Epsilon')
plt.show()


# %%
plt.figure()
plt.scatter(dt1['Strike'], dt1['Mid'], c='b', marker='o', s=15, alpha=0.3)
plt.scatter(dt2['Strike'], dt2['Mid'], c='r', marker='o', s=10, alpha=0.3)
plt.scatter(dt3['Strike'], dt3['Mid'], c='g', marker='o', s=10, alpha=0.3)
plt.title("TSLA1, Mat = 3 Days, Before")
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.legend(labels=['t1', 't2', 't3'])
plt.xlim(1.3,1.5)
plt.ylim(0,0.01)
# plt.xlim(1.3, 1.5)
# plt.ylim(0.0004, 0.002)
plt.show()
#%%
plt.figure()
plt.scatter(dt1['Strike'], dt1['Mid']+e[0:n[0]], c='b', marker='o', s=15, alpha = 0.3)
plt.scatter(dt2['Strike'], dt2['Mid']+e[n[0]:(n[0]+n[1])], c='r', marker='o', s=10, alpha=0.3)
plt.scatter(dt3['Strike'], dt3['Mid']+e[(n[0]+n[1]):(n[0]+n[1]+n[2])], c='g', marker='o', s=10, alpha=0.3)
plt.title("TSLA1, Mat = 3 Day, After")
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.legend(labels=['t1', 't2', 't3'])
plt.xlim(1.3,1.5)
plt.ylim(0,0.01)
plt.show()

4#%%
plt.figure()
plt.scatter(dt1['Strike'], dt1['Mid'], c='b', marker='o', s=10, alpha=0.3)
plt.scatter(dt1['Strike'], dt1['Mid']+e[0:n[0]], c='r', marker='x', s=10, alpha = 0.3)
plt.legend(labels=['Raw', 'Repaired'])
plt.title("TSLA1, Mat = 3 Days")
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.xlim(1.3, 1.5)
plt.ylim(0.0004, 0.002)
plt.show()
#%%