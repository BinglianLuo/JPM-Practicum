import pandas as pd
from scipy import optimize
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
import time
import seaborn as sns
from scipy.optimize import fmin_bfgs


# BSM Option Price Calculation
def bs_price(cp_flag, s, k, t, r, v, q=0.0):
    n = norm.pdf
    N = norm.cdf
    d1 = (np.log(s / k) + (r + v * v / 2.) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if cp_flag == 'c':
        price = s * np.exp(-q * t) * N(d1) - k * np.exp(-r * t) * N(d2)
    else:
        price = k * np.exp(-r * t) * N(-d2) - s * np.exp(-q * t) * N(-d1)
    return price


# Implied Vol Calculation
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


# B* func Calculation
def Bfunc(S, K, t, sigma):
    inside = (K - S) / (np.sqrt(2 * t) * sigma)
    outsideup = sigma * np.sqrt(t) * np.exp(-(K - S) ** 2 / (2 * t * sigma ** 2))
    outsidedown = np.sqrt(2 * np.pi)
    return 0.5 * (S - K) * (1 - erf(inside)) + outsideup / outsidedown


# Part 3.1: Optimization over sigma0
def loss_func(s, t, k, c):
    # S: constant spot price
    # T: constant and calculated
    # K: list, imported
    # c: list, imported
    def find_sigma(x):
        a = 0
        for i in range(0, len(k)):
            a += (Bfunc(s, k[i], t, x) - c[i]) ** 2
        return a

    return optimize.fmin(find_sigma, 10, disp=False)


# Market Days Calculation
def mkt_time(t1, t2):
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=t1, end_date=t2)
    t = mcal.date_range(early, frequency='1D')
    return len(t) / 252


# Iu Calculation
def cal_Iu(V, h):
    A = (K[0] - S0) / sigma
    B = (K[1] - S0) / sigma
    sum = 0.5 * np.exp((-h * sigma) ** 2 / 2) * \
          (erf((B + h * sigma) / np.sqrt(2)) - erf((A + h * sigma) / np.sqrt(2)))
    # print(sum)
    for k in range(0, len(dt['Strike'])):
        A = (K[k + 1] - S0) / sigma
        B = (K[k + 2] - S0) / sigma
        alpha = -V[0:k + 1].sum() - h
        beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0

        sum += 0.5 * np.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0) * \
               (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2)))
        # print(alpha, beta, beta + (alpha * sigma) ** 2 / 2 + alpha * S0,
        #       np.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0),
        #       (B - alpha * sigma) / np.sqrt(2), (A - alpha * sigma) / np.sqrt(2), sum)
    return sum


# Ih Calculation
# Solve for h
# Without using the value of u
def solve_Ih(V):
    def Ih(h):
        A = (K[0] - S0) / sigma
        B = (K[1] - S0) / sigma
        alpha = - h
        beta = h * S0
        sum = np.exp(beta) * \
              (2 * sigma * np.exp(alpha * S0)
               * (np.exp(A * alpha * sigma - A ** 2 / 2) - np.exp(B * alpha * sigma - B ** 2 / 2)) +
               np.sqrt(2 * np.pi) * alpha * sigma ** 2 * np.exp((alpha * sigma) ** 2 / 2 + alpha * S0) *
               (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))

        for k in range(0, len(dt['Strike'])):
            A = (K[k + 1] - S0) / sigma
            B = (K[k + 2] - S0) / sigma
            alpha = -V[0:k + 1].sum() - h
            beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
            sum += np.exp(beta) * \
                   (2 * sigma * np.exp(alpha * S0)
                    * (np.exp(A * alpha * sigma - A ** 2 / 2) - np.exp(B * alpha * sigma - B ** 2 / 2)) +
                    np.sqrt(2 * np.pi) * alpha * sigma ** 2 * np.exp((alpha * sigma) ** 2 / 2 + alpha * S0) *
                    (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))
        return sum

    root = optimize.brentq(Ih, -0.01, 0.01)
    return root


# Ik Calculation
def Ik(h, V, K):
    V_k = np.zeros(len(V))
    for j in range(0, len(V)):
        sum = 0
        for k in range(j, len(dt['Strike'])):
            A = (K[k + 1] - S0) / sigma
            B = (K[k + 2] - S0) / sigma
            alpha = -V[0:k + 1].sum() - h
            beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
            sum += (2 * sigma * np.exp(alpha * S0) \
                    * (np.exp(A * alpha * sigma - A ** 2 / 2 + beta) - np.exp(B * alpha * sigma - B ** 2 / 2)) + \
                    np.sqrt(2 * np.pi) * (alpha * sigma ** 2 - K[j + 1] + S0) * np.exp(
                        (alpha * sigma) ** 2 / 2 + alpha * S0 + beta) * \
                    (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))
        V_k[j] = sum / (2 * np.sqrt(2 * np.pi))
    return V_k


# f1 partial calculation
# Used for g1 partial
def f1_partial(V):
    partial = np.zeros(len(V))
    for i in range(0, len(V)):
        if delta_bid[i] <= V[i] * w[i] <= delta_ask[i]:
            partial[i] = V[i] * w[i]
        elif V[i] * w[i] > delta_ask[i]:
            partial[i] = delta_ask[i]
        else:
            partial[i] = delta_bid[i]
    return partial


# g1 partial calculation
# Not used in DE
def g1_partial(V, u, h):
    return f1_partial(V) + c_mid - Ik(h, V, K) * np.exp(-u)


# Sum of f1
# Used to calculate g1
def sum_f1(V):
    f = [0 for x in range(0, len(c_bid))]
    for i in range(0, len(f)):
        if (V[i] * w[i] >= delta_bid[i]) and (V[i] * w[i] <= delta_ask[i]):
            f[i] = V[i] ** 2 * w[i] / 2
        elif V[i] * w[i] > delta_ask[i]:
            f[i] = delta_ask[i] * V[i] - delta_ask[i] ** 2 / (2 * w[i])
        else:
            f[i] = delta_bid[i] * V[i] - delta_bid[i] ** 2 / (2 * w[i])
    return np.sum(f)


# g1 Calculation
def g1(u, h, V):
    res = u + sum_f1(V) + np.sum(V * c_mid) + cal_Iu(V, h) * np.exp(-u)
    return res[0]


# DE to update V
def V_update(u, h):
    def solve_V(V):
        return g1(u, h, V)

    bounds = [(-0.0018, 0.0018)] * 26
    result = differential_evolution(solve_V, bounds)
    v = result.x
    return v


# print(fmin_bfgs(g1, np.ones(len(dt['Strike'])) * 0.000001, fprime=g1_prime))


# Call price calculation
def call_price(u, h, V):
    # Remark4.3
    def alphafuncKlast(alpha, sigma, K1, K2, S0, K):
        A = (K1 - S0) / sigma
        B = (K2 - S0) / sigma
        out = np.exp(alpha * S0) / (2 * np.sqrt(2 * np.pi))
        one = 2 * sigma * np.exp(A * alpha * sigma - A ** 2 / 2)
        two = np.sqrt(2 * np.pi) * np.exp(alpha ** 2 * sigma ** 2 / 2) \
              * erf((A - alpha * sigma) / np.sqrt(2)) * (alpha * sigma ** 2 - K + S0)
        three = np.sqrt(2 * np.pi) * np.exp(alpha ** 2 * sigma ** 2 / 2) \
                * erf((B - alpha * sigma) / np.sqrt(2)) * (alpha * sigma ** 2 - K + S0)
        four = 2 * sigma * np.exp(B * alpha * sigma - B ** 2 / 2)
        return (out * (one - two + three - four))

    integro = np.zeros(len(dt['Strike']))
    Kmax = 10000
    for n in range(len(dt['Strike'])):
        K = np.append(dt['Strike'][n:], Kmax)
        for i in range(len(K) - 1):
            outside = np.exp(h * S0 - u + V[0] * K[0])
            alpha = -h - V[0]
            # print(call_strike[i],call_strike[i+1])
            for j in range(1, i + 1):
                outside *= np.exp(V[j] * K[j])
                alpha -= V[j]
            integro[n] += outside * alphafuncKlast(alpha, sigma, K[i], K[i + 1], S0, K[0])

    # first = df+call_mid-integro
    return (integro * np.exp(-u))


print("Complete")
# %%
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/AMZNtest1.xlsx')
# V = np.array(pd.read_excel('V.xlsx', header = None))[:,0]
K = np.append(0, dt['Strike'])
K = np.append(K, 10000)
V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))*0.000001
u = 0
h = 0
S0 = 2720.29
t = mkt_time('2022-03-08', '2022-06-17')
dt['Bid_IV'] = 0
dt['Ask_IV'] = 0
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
dt['Moneyness'] = 0
sigma = loss_func(2720.29, t, dt['Strike'], dt['Mid']) * np.sqrt(t)
c_bid = dt['Bid']
c_ask = dt['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]
# V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))*0.001
# V = np.array(pd.read_excel('V2.xlsx', header = None))[:,0]

print("Complete")
# %%
D = np.eye(len(dt['Strike']))
a = 0.00001
l = []
start_time = time.time()
print(g1_partial(V, u, h))
epsilon = np.linalg.norm(g1_partial(V, u, h))
i = 0

for i in range(0, 100):
    # while epsilon > 0.001:
    g1_1 = g1_partial(V, u, h)
    d = -D @ np.transpose([g1_1])
    s = a * d
    V_update = V + s.T[0]
    g1_2 = g1_partial(V_update, u, h)
    epsilon = np.linalg.norm(g1_2)
    # epsilon = np.linalg.norm(d)
    l.append(epsilon)
    print(i, epsilon)
    if epsilon > 0.0001:
        y = g1_2 - g1_1
        D = (np.eye(len(dt['Strike'])) - (s @ np.array([y])) / (y @ s)) @ D @ (
                np.eye(len(dt['Strike'])) - (np.array([y]).T @ s.T) / (y @ s)) \
            + (s @ s.T) / (y @ s)
        V = V_update
        i = i + 1
    else:
        break
    print(g1_partial(V_update, u, h))
# print(V)
print("--- %s seconds ---" % (time.time() - start_time))


# %% Implementation sample
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/AMZNtest1.xlsx')
# V = np.array(pd.read_excel('V.xlsx', header = None))[:,0]
K = np.append(0, dt['Strike'])
K = np.append(K, 10000)
V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))*0.000001
u = 0
h = 0
S0 = 2720.29
t = mkt_time('2022-03-08', '2022-06-17')
dt['Bid_IV'] = 0
dt['Ask_IV'] = 0
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
dt['Moneyness'] = 0
sigma = loss_func(2720.29, t, dt['Strike'], dt['Mid']) * np.sqrt(t)

c_bid = dt['Bid']
c_ask = dt['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]

u_l = []
h_l = []
g1_l = []
for i in range(0, 1000):
    start_time = time.time()
    u = np.log(cal_Iu(V, h))[0]
    u_l.append(u)
    h = solve_Ih(V)
    h_l.append(h)
    V = V_update(u, h)
    func_val = g1(u, h, V)
    g1_l.append(func_val)
    # print(V)

    print("Iteration %s --- %s seconds ---" % (i + 1, time.time() - start_time))
    print("G1 = ", func_val)
    print("u = %s , h = %s" % (u, h))
    print(np.round(V, 6))


# %% Iteration Visualization
# plt.plot(u_l)
# plt.plot(h_l)
# plt.show()

plt.plot(g1_l)
plt.legend("g1")
plt.title("Iteration Times = 1000, Boundary = ±0.001")
plt.show()

# pd.DataFrame(V).to_excel('DE_V.xlsx')


# %% Model visualization
dt['Model'] = call_price(u, h, V)
for i in range(0, len(dt)):
    dt.loc[i, 'Moneyness'] = dt.loc[i, 'Strike'] / S0
    dt.loc[i, 'Bid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Bid'])
    dt.loc[i, 'Ask_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Ask'])
    dt.loc[i, 'Mid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Mid'])
    dt.loc[i, 'Model_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model'])

dt = dt[~dt['Bid_IV'].isin([0])]
# dt = dt[~dt['Model_IV'].isin([0])]
plt.scatter(dt['Moneyness'], dt['Ask_IV'], c='blue', marker='o', s=10)
plt.scatter(dt['Moneyness'], dt['Bid_IV'], c='orange', marker='^', s=10)
plt.plot(dt['Moneyness'], dt['Mid_IV'], c='purple')
plt.scatter(dt['Moneyness'], dt['Model_IV'], c='red', marker='x', s=10)
# plt.xlim(0.75, 1.2)
# plt.ylim(0.3, 0.6)
plt.legend(labels=['Ask', 'Bid', 'Mid', 'Model'])
plt.title("Iteration Times = 1000, Boundary = ±0.001")
plt.show()
