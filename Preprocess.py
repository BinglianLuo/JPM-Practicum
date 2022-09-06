import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from scipy.stats import norm
from scipy import optimize
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import time


def mkt_time(t1, t2):
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=t1, end_date=t2)
    t = mcal.date_range(early, frequency='1D')
    return (len(t) + 1) / 252


def bs_price(cp_flag, s, k, t, r, v, q=0.0):
    n = norm.pd
    N = norm.cdf
    d1 = (np.log(s / k) + (r + v * v / 2.) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if cp_flag == 'c':
        price = s * np.exp(-q * t) * N(d1) - k * np.exp(-r * t) * N(d2)
    else:
        price = k * np.exp(-r * t) * N(-d2) - s * np.exp(-q * t) * N(-d1)
    return price

def iv_cal(cp_flag, s, k, t, r, target):
    # Need to add put formula
    def fit(sigma):
        return bs_price(cp_flag, s, k, t, r, sigma, q=0.0) - target

    if target >= s - k * np.exp(-r * t):
        # initial = np.sqrt(2 * np.pi / t) * target / s
        root = optimize.brentq(fit, -1, 6)
        return root
    else:
        print("No solution")
        return 0


#%%
date_list = list(pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/date_list.xlsx', header=None)[0])[0:5]
start_date = '2022-01-25'

dt = []
t = []
n = []
k = []
c_ask = []
c_bid = []
c_model = []
c_mid = []
date_select_index = []

for i in range(0, len(date_list)):
    dt.append(pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/Model_ManyHands.xlsx', sheet_name=date_list[i]))

for i in range(0, len(date_list)):
    t.append(mkt_time(start_date, date_list[i]))
    n.append(len(dt[i]))
    k.append(list(dt[i]['Strike']))
    c_bid += list(dt[i]['Bid'])
    c_ask += list(dt[i]['Ask'])
    c_model += list(dt[i]['Model'])
    c_mid += list(dt[i]['Mid'])
    date_select_index.append(i)


N = sum(n)
m = len(n)
delta_bid = [(b - a) for a, b in zip(c_bid, c_model)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_model)]
delta0 = min(1 / N, min(delta_ask), min(delta_bid))
print("Complete")

# %%
A = np.zeros(shape=(0, 2 * N))
b = []
# C1 Outright: m
for i in range(0, m):
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:(i + 1)]) - 1] = 1
    A = np.append(A, row, axis=0)
    # A = np.r_[A, row]
    b.append(0)

print("Complete C1")

#%%
# C2 Vertical spread: N + m
A1 = np.zeros(shape=(0, 2 * N))
for i in range(0, m):
    # VS_0 >= 0, -c1 >= -1
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:i])] = -1
    A1 = np.r_[A1, row]
    b.append(-1)

    # VS_0 <= 1, c1 >= 1 - k1
    row = np.zeros(shape=(1, 2 * N))
    row[0][sum(n[0:i])] = 1
    A1 = np.r_[A1, row]
    b.append(1 - k[i][0])

    # VS_j >= 0, c1 - c2 >= 0
    for j in range(0, n[i] - 1):
        row = np.zeros(shape=(1, 2 * N))
        row[0][sum(n[0:i]) + j] = 1
        row[0][sum(n[0:i]) + j + 1] = -1
        A1 = np.r_[A1, row]
        b.append(0)

A = np.append(A, A1, axis=0)
print("Complete C2")

# %% C3 Vertical butterfly: N - m
start_time = time.time()
A1 = np.zeros(shape=(0, 2 * N))
for i in range(0, m):
    for j in range(0, n[i] - 1):
        # -k2*c1 + k1*c2 >= k1 - k2
        if j == 0:
            row = np.zeros(shape=(1, 2 * N))
            row[0][sum(n[0:i])] = -k[i][1]
            row[0][sum(n[0:i]) + 1] = k[i][0]
            A1 = np.r_[A1, row]
            b.append(k[i][0] - k[i][1])

        # (k3-k2)c1 + (k1-k3)c2 + (k2-k1)c3 >= 0
        else:
            row = np.zeros(shape=(1, 2 * N))
            row[0][sum(n[0:i]) + j - 1] = -k[i][j] + k[i][j + 1]
            row[0][sum(n[0:i]) + j] = k[i][j - 1] - k[i][j + 1]
            row[0][sum(n[0:i]) + j + 1] = k[i][j] - k[i][j - 1]
            A1 = np.r_[A1, row]
            b.append(0)

A = np.append(A, A1, axis=0)
print("Complete C3")
print("Total time is %s seconds ---" % (time.time() - start_time))

#%%
# C4 Calendar spread
start_time = time.time()
A1 = np.zeros(shape=(0, 2 * N))
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
                    A1 = np.r_[A1, row]
                    A_count += 1
                    b.append(0)
                    # print(i1, j1, k[i1][j1])
                    # print(i2, j2, k[i2][j2])
                    break

A = np.append(A, A1, axis=0)
print("Compute_count = ", compute_count)
print("Constraint_count = ", A_count)
print("Complete C4")
print("Total time is %s seconds ---" % (time.time() - start_time))
del i, j, i1, i2, j1, j2


# %%
A1 = np.zeros(shape=(0, 2 * N))
for i1 in range(0, m):
    for i2 in range(i1, m):
        for j1 in range(0, n[i1]):
            for j2 in range(0, n[i2]):
                if k[i1][j1] >= k[i2][j2]:
                    # print(i1, i2, j1, j2)
                    row = np.zeros(shape=(1, 2 * N))
                    row[0][sum(n[0:i1])] = -1
                    row[0][sum(n[0:i2])] = 1
                    A1 = np.r_[A1, row]
                    b.append(0)
                    break
A = np.append(A, A1, axis=0)

print("Complete C5")

# %%
A1 = np.zeros(shape=(0, 2 * N))
for i1 in range(0, m):
    for i2 in range(0, m):
        for j1 in range(0, n[i1]):
            for j2 in range(0, n[i2]):
                for i in range(min(i1, i2)):
                    for j in range(0, n[i]):
                        if k[i1][j1] < k[i][j] < k[i2][j2]:
                            row = np.zeros(shape=(1, 2 * N))
                            row[0][sum(n[0:i1])] = k[i2][j2] - k[i][j]
                            row[0][sum(n[0:i2])] = k[i][j] - k[i1][j1]
                            row[0][sum(n[0:i])] = k[i2][j2] - k[i1][j1]
                            A1 = np.r_[A1, row]
                            b.append(0)
                    break
A = np.append(A, A1, axis=0)

print("Complete C6")

#%%
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

A = unique_rows(A)
#%% Arb-count

sum(A[:, :N] @ c_model >= b)/len(A)
# sum(A[:, :N] @ c_repaired >= b)/len(A)

# %%
c_lp = c_model + [0] * N
b_lp = list(-matrix(b) + matrix(A) * matrix(c_lp))
A_lp = -A
obj = [0] * N + [1] * N


# c = c_model + [0] * N
# b = list(-matrix(b) + matrix(A) * matrix(c))
# A = -A
# obj = [0] * N + [1] * N


# %%
A1 = np.zeros(shape=(0, 2 * N))
start_time = time.time()

for j in range(0, N):
    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = -1
    row[0][N + j] = -1
    A1 = np.append(A1, row, axis=0)
    # A = np.r_[A, row]
    b_lp.append(delta_bid[j] - delta0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = 1
    row[0][N + j] = -1
    A1 = np.append(A1, row, axis=0)
    # A = np.r_[A, row]
    b_lp.append(delta_ask[j] - delta0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = -delta0 / delta_bid[j]
    row[0][N + j] = -1
    A1 = np.append(A1, row, axis=0)
    # A = np.r_[A, row]
    b_lp.append(0)

    row = np.zeros(shape=(1, 2 * N))
    row[0][j] = delta0 / delta_ask[j]
    row[0][N + j] = -1
    A1 = np.append(A1, row, axis=0)
    # A = np.r_[A, row]
    b_lp.append(0)

A_lp = np.append(A_lp, A1, axis=0)
print("Complete C")
print("Total time is %s seconds ---" % (time.time() - start_time))



# %%
obj = matrix(obj, (len(obj), 1), 'd')
b_lp = matrix(b_lp, (len(b_lp), 1), 'd')

sol = solvers.lp(matrix(obj), matrix(A_lp), matrix(b_lp))

# %%
e = [round(num, 8) for num in list(sol['x'])]
# e = list(sol['x'])
print(1 - e[0:N].count(0) / N)
c_repaired = [(a + b) for a, b in zip(c_lp, e)][0:N]
# c_before = c[0:N]

#%%
def deviation_ratio(c_mid, c_model, c_ask, c_bid):
    sum_value = np.sum(np.abs(np.array(c_mid) - np.array(c_model)) / (np.array(c_ask) - np.array(c_bid)) / len(c_mid))
    return sum_value

print(deviation_ratio(c_mid, c_repaired, c_ask, c_bid))

# %% Error

# plt.scatter([i for i in range(n[0])], e[0:n[0]], c='b', marker='o', s=10, alpha=0.5)
# plt.axhline(y=0, color='r', linestyle='-', alpha = 0.5)
# plt.title("Call Price Perturbation Value")
# plt.ylabel('Epsilon')
# plt.show()


# %%
# plt.figure()
# plt.scatter(dt1['Strike'], dt1['Mid'], c='b', marker='o', s=15, alpha=0.3)
# plt.scatter(dt2['Strike'], dt2['Mid'], c='r', marker='o', s=10, alpha=0.3)
# plt.scatter(dt3['Strike'], dt3['Mid'], c='g', marker='o', s=10, alpha=0.3)
# plt.title("TSLA1, Mat = 3 Days, Before")
# plt.xlabel('Strike')
# plt.ylabel('Call price')
# plt.legend(labels=['t1', 't2', 't3'])
# plt.xlim(1.3,1.5)
# plt.ylim(0,0.01)
# # plt.xlim(1.3, 1.5)
# # plt.ylim(0.0004, 0.002)
# plt.show()
# %%
# plt.figure()
# plt.scatter(dt1['Strike'], dt1['Mid']+e[0:n[0]], c='b', marker='o', s=15, alpha = 0.3)
# plt.scatter(dt2['Strike'], dt2['Mid']+e[n[0]:(n[0]+n[1])], c='r', marker='o', s=10, alpha=0.3)
# plt.scatter(dt3['Strike'], dt3['Mid']+e[(n[0]+n[1]):(n[0]+n[1]+n[2])], c='g', marker='o', s=10, alpha=0.3)
# plt.title("TSLA1, Mat = 3 Day, After")
# plt.xlabel('Strike')
# plt.ylabel('Call price')
# plt.legend(labels=['t1', 't2', 't3'])
# plt.xlim(1.3,1.5)
# plt.ylim(0,0.01)
# plt.show()

# %%
plt.figure()
plt.scatter(dt[2]['Strike'], dt[2]['Mid'], c='b', marker='o', s=10, alpha=0.3)
# plt.scatter(dt1['Strike'], dt1['Mid']+e[0:n[0]], c='r', marker='x', s=10, alpha = 0.3)
# plt.legend(labels=['Raw', 'Repaired'])
plt.title("TSLA1, Mat = 3 Days")
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.xlim(1.4, 1.6)
plt.ylim(0.000, 0.01)
plt.show()


#%%
for i in range(0, m):
    dt[i]['Model'] = c_model[sum(n[0:i]):sum(n[0:(i+1)])]
#%%
plt.figure()
i = 0
plt.scatter(dt[i]['Strike'], dt[i]['Mid'], c='b', marker='o', s=10, alpha=0.3)
# plt.scatter(dt1['Strike'], dt1['Mid']+e[0:n[0]], c='r', marker='x', s=10, alpha = 0.3)
# plt.legend(labels=['Raw', 'Repaired'])
plt.title("TSLA1, Mat = 3 Days")
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.xlim(1.32, 1.48)
# plt.ylim(0.003, 0.008)
plt.show()


# %%
d = pd.DataFrame()
d['Model'] = c_repaired[0:N]
d['Maturity'] = 0
d['Model_IV'] = 0
Normalized = []
Strike = []

for i in range(0, m):
    d['Maturity'][sum(n[0:i]):sum(n[0:(i+1)])] = t[i]
    Normalized += list(dt[i]['Normalized'])
    Strike += list(dt[i]['Strike'])

d['Normalized'] = Normalized
d['Strike'] = Strike
# d['Mid'] = c_mid

for i in range(0, len(d)):
    d.loc[i, 'Model_IV'] = iv_cal('c', 1, d.loc[i, 'Strike'], d.loc[i, 'Maturity'], 0, d.loc[i, 'Model'])

del Normalized, Strike, i
#
# d = d[d["Normalized"] >= -1]
# d = d[d["Normalized"] <= 1.5]
d = d[d["Strike"] >= 0.5]
d = d[d["Strike"] <= 1.5]


x = d['Strike']
y = d['Maturity']
z = d['Model_IV']

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')


ax.plot_trisurf(x, y, z, cmap=cm.inferno)
ax.set(xlabel='Strike', ylabel='Maturity', zlabel='Implied Volatility', title='Model4: LP + Sinkhorn P12 + LP')
ax.tick_params(axis='both', which='major', labelsize=8)

max_yticks = 5
yloc = plt.MaxNLocator(max_yticks)
ax.yaxis.set_major_locator(yloc)
fig.show()
#%%
# plt.scatter(d['Normalized'][-44:], d['Model_IV'][-44:], c='b', marker='o', s=10)
plt.scatter(dt[15]['Normalized'], c_model[-44:], c='b', marker='o', s=10)
plt.title("TSLA1, Mat = 2022-12-16")
plt.xlabel('Normalized Strike')
plt.ylabel('Model Call Price')
plt.ylim(0,0.6)
plt.xlim(-1,1.1)
#%%
plt.scatter(dt[15]['Normalized'][-44:], d['Model_IV'][-44:], c='crimson', marker='X', s=15)
plt.title("TSLA1, Mat = 2022-12-16")
plt.xlabel('Normalized Strike')
plt.ylabel('Implied Volatility')
plt.xlim(-1,1.1)

#%%
from openpyxl import load_workbook

path = r'/Users/binglianluo/Desktop/Spring2022/Practicum/Model4.xlsx'
for i in range (0, len(date_list)):

    book = load_workbook(path)

    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = book

    d = dt[i]
    d.to_excel(writer, sheet_name=date_list[i], index=None)

    writer.save()
