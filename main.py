import pandas as pd
from scipy import optimize
from scipy.stats import norm
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
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


# B* func
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

    return optimize.fmin(find_sigma, 10)

# print(loss_func(566.82, 3/252, dt['Strike'],dt['Mid']))

# Market Days Calculation
def time(t1, t2):
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=t1, end_date=t2)
    t = mcal.date_range(early, frequency='1D')
    return len(t) / 252


# %% Iu Calculation
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLAtest3.xlsx')
K = np.append(0, dt['Strike'])
K = np.append(K, 10000)
V = np.zeros(len(dt['Strike']))
u = 0
h = 0
S0 = 566.82
sigma = 701.8359375 * np.sqrt(3 / 252)


def cal_Iu(V, h):
    A = (K[0] - S0) / sigma
    B = (K[1] - S0) / sigma
    sum = np.exp(h * S0) * 0.5 * np.exp((-h * sigma) ** 2 / 2 - h * S0) * \
          (erf((B + h * sigma) / np.sqrt(2)) - erf((A + h * sigma) / np.sqrt(2)))
    for k in range(0, len(dt['Strike'])):
        A = (K[k + 1] - S0) / sigma
        B = (K[k + 2] - S0) / sigma
        alpha = -V[0:k + 1].sum() - h
        beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
        sum += np.exp(beta) * 0.5 * np.exp((alpha * sigma) ** 2 / 2 + alpha * S0) * \
               (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2)))
        # print(K[k + 1], sum)
    return sum


print(cal_Iu(V, h))


# %% Ih Calculation

def solve_Ih(V):
    def Ih(h):
        A = (K[0] - S0) / sigma
        B = (K[1] - S0) / sigma
        alpha = - h
        beta = h * S0
        sum = np.exp(beta) * \
              (2 * sigma * np.exp(alpha * S0) \
               * (np.exp(A * alpha * sigma - A ** 2 / 2) - np.exp(B * alpha * sigma - B ** 2 / 2)) + \
               np.sqrt(2 * np.pi) * alpha * sigma ** 2 * np.exp((alpha * sigma) ** 2 / 2 + alpha * S0) * \
               (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))
        # print (sum)
        for k in range(0, len(dt['Strike'])):
            A = (K[k + 1] - S0) / sigma
            B = (K[k + 2] - S0) / sigma
            alpha = -V[0:k + 1].sum() - h
            beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
            sum += np.exp(beta) * \
                   (2 * sigma * np.exp(alpha * S0) \
                    * (np.exp(A * alpha * sigma - A ** 2 / 2) - np.exp(B * alpha * sigma - B ** 2 / 2)) + \
                    np.sqrt(2 * np.pi) * alpha * sigma ** 2 * np.exp((alpha * sigma) ** 2 / 2 + alpha * S0) * \
                    (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))
            # print(sum / (2 * np.sqrt(2 * np.pi)))
        return sum / (2 * np.sqrt(2 * np.pi))

    root = optimize.brentq(Ih, -0.01, 0.01)
    return root
    # return Ih(0)


print(solve_Ih(V))


# %%
# V = V/1000
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
            # print(sum)
        V_k[j] = sum / (2 * np.sqrt(2 * np.pi))
    return V_k


print(Ik(0, V, K))

# %%
c_bid = dt['Bid']
c_ask = dt['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]


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


def g1_partial(V, u, h):
    return f1_partial(V) + c_mid - Ik(h, V, K) * np.exp(-u)


# print(Ik(0, V, K)*np.exp(-u))
# print(f1_partial(V)+c_mid-Ik(0, V, K)*np.exp(-u))

print(g1_partial(V, u, h))


# %%
# def sum_f1(V):
#     f = [0 for x in range(0, len(c_bid))]
#     for i in range(0, len(f)):
#         if (V[i] * w[i] >= delta_bid[i]) and (V[i] * w[i] <= delta_ask[i]):
#             f[i] = V[i] ** 2 * w[i] / 2
#         elif V[i] * w[i] > delta_ask[i]:
#             f[i] = delta_ask[i] * V[i] - delta_ask[i] ** 2 / (2 * w[i])
#         else:
#             f[i] = delta_bid[i] * V[i] - delta_bid[i] ** 2 / (2 * w[i])
#     return sum(f)
#
#
# def g1(V):
#     u = 0
#     res = u + sum_f1(V) + sum(V * c_mid) + Iu(h, V, K) * np.exp(-u)
#     return res
#
#
# def g1_prime(V):
#     return g1_partial(V, 0, 0)
#
#
# print(fmin_bfgs(g1, np.zeros(len(dt['Strike']), fprime=g1_prime)))

# %% Update of V
u = 0
h = 0
# V = np.zeros(len(dt['Strike']))
V = np.zeros(len(dt['Strike']))
D = np.eye(len(dt['Strike']))
a = 0.0001
l = []

from scipy.optimize import fmin_bfgs

for i in range(0, 10):
    # while 1:
    d = -D @ np.transpose([g1_partial(V, u, h)])
    s = a * d
    V_update = V + s.T[0]
    # print(V_update)
    epsilon = np.linalg.norm(g1_partial(V_update, u, h))
    l.append(epsilon)
    print(epsilon)
    if epsilon > 0.0001:
        y = g1_partial(V_update, u, h) - g1_partial(V, u, h)
        # D = D + (s @ s.T) / (s.T @ y) - (D @ np.transpose([y]) @ np.array([y]) @ D) / (np.array([y]) @ D @
        # np.transpose([y]))
        D = (np.eye(len(dt['Strike'])) - (s @ np.array([y])) / (y @ s)) @ D @ (
                np.eye(len(dt['Strike'])) - (np.array([y]).T @ s.T) / (y @ s)) \
            + (s @ s.T) / (y @ s)
        V = V_update
    else:
        break
print(V)


# %%
def cpfgVlast(V):
    #########
    u = 0
    h = 0

    #########

    # Remark4.3
    def alphafuncKlast(alpha, sigma, K1, K2, S0, K):
        A = (K1 - S0) / sigma;
        B = (K2 - S0) / sigma;
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
    return (integro)


print(cpfgVlast(V))

# %% Implementation sample
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLAtest3.xlsx')
s = 566.82
t = 3 / 252

dt['Bid_IV'] = 0
dt['Ask_IV'] = 0
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
dt['Moneyness'] = 0
dt['Model'] = cpfgVlast(V)
for i in range(0, len(dt)):
    dt.loc[i, 'Moneyness'] = s / dt.loc[i, 'Strike']
    dt.loc[i, 'Bid_IV'] = iv_cal('c', s, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Bid'])
    dt.loc[i, 'Ask_IV'] = iv_cal('c', s, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Ask'])
    dt.loc[i, 'Mid_IV'] = iv_cal('c', s, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Mid'])
    dt.loc[i, 'Model_IV'] = iv_cal('c', s, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model'])

dt = dt[~dt['Bid_IV'].isin([0])]
plt.scatter(dt['Moneyness'], dt['Ask_IV'], c='blue', marker='o', s=10)
plt.scatter(dt['Moneyness'], dt['Bid_IV'], c='orange', marker='^', s=10)
plt.plot(dt['Moneyness'], dt['Mid_IV'], c='purple')
plt.plot(dt['Moneyness'], dt['Model_IV'], c='red')
plt.ylim(1, 1.6)
plt.legend(labels=['Ask', 'Bid', 'Mid', 'Model'])
plt.show()
