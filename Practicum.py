import pandas as pd
from scipy import optimize
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
import time
import sympy as sym
import sklearn.metrics
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
    return (len(t) + 1) / 252


# Iu Calculation
def cal_Iu(V, h):
    A = (K[0] - S0) / sigma
    B = (K[1] - S0) / sigma
    sum = 0.5 * np.exp((-h * sigma) ** 2 / 2) * \
          (erf((B + h * sigma) / np.sqrt(2)) - erf((A + h * sigma) / np.sqrt(2)))
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


def cal_Iu1(V, h):
    A = (K[0] - S0) / sigma
    B = (K[1] - S0) / sigma
    sum = 0.5 * np.exp((-h * sigma) ** 2 / 2) * \
          (erf((B + h * sigma) / np.sqrt(2)) - erf((A + h * sigma) / np.sqrt(2)))
    for k in range(0, len(dt['Strike'])):
        A = (K[k + 1] - S0) / sigma
        B = (K[k + 2] - S0) / sigma
        alpha = -V[0:k + 1].sum() - h
        beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
        sum += 0.5 * sym.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0) * \
               (sym.erf((B - alpha * sigma) / np.sqrt(2)) - sym.erf((A - alpha * sigma) / np.sqrt(2)))
        # print(alpha, beta, beta + (alpha * sigma) ** 2 / 2 + alpha * S0,
        #       np.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0),
        #       (B - alpha * sigma) / np.sqrt(2), (A - alpha * sigma) / np.sqrt(2), sum)
    return sym.N(sum)


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

    root = optimize.brentq(Ih, -10, 10)
    return root


def solve_Ih1(V):
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

    root = optimize.fsolve(Ih, 0)
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
            sum += (2 * sigma * np.exp(alpha * S0 + beta) \
                    * (np.exp(A * alpha * sigma - A ** 2 / 2) - np.exp(B * alpha * sigma - B ** 2 / 2)) + \
                    np.sqrt(2 * np.pi) * (alpha * sigma ** 2 - K[j + 1] + S0) * np.exp(
                        (alpha * sigma) ** 2 / 2 + alpha * S0 + beta) * \
                    (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2))))
        V_k[j] = sum / (2 * np.sqrt(2 * np.pi))
    return V_k


def Ik1(h, V, K):
    V_k = sym.zeros(1, len(V))
    for j in range(0, len(V)):
        sum = 0
        for k in range(j, len(dt['Strike'])):
            A = (K[k + 1] - S0) / sigma
            B = (K[k + 2] - S0) / sigma
            alpha = -V[0:k + 1].sum() - h
            beta = (V[0:k + 1] * K[1:k + 2]).sum() + h * S0
            sum += sym.N((2 * sigma * sym.exp(alpha * S0 + beta) \
                          * (sym.exp(A * alpha * sigma - A ** 2 / 2) - sym.exp(B * alpha * sigma - B ** 2 / 2)) + \
                          sym.sqrt(2 * sym.pi) * (alpha * sigma ** 2 - K[j + 1] + S0) * sym.exp(
                        (alpha * sigma) ** 2 / 2 + alpha * S0 + beta) * \
                          (sym.erf((B - alpha * sigma) / np.sqrt(2)) - sym.erf((A - alpha * sigma) / np.sqrt(2)))))
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
def g1_partial1(V, u, h):
    return sym.Array(f1_partial(V) + c_mid - sym.Array(Ik1(h, V, K) * np.exp(-u))[0])


def g1_partial(V, u, h):
    return f1_partial(V) + c_mid - Ik(h, V, K) * np.exp(-u)


def sp_norm(a):
    s = 0
    for i in range(0, len(a)):
        s += a[i] ** 2
    return sym.sqrt(s)


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
    res = u + sum_f1(V) + np.sum(V * c_mid) + cal_Iu(V, h) * np.exp(float(-u))
    return res


def g11(u, h, V):
    res = u + sum_f1(V) + np.sum(V * c_mid) + cal_Iu1(V, h) * np.exp(float(-u))
    return res


# DE to update V
def V_update(u, h, V, bound):
    l = len(V)

    def solve_V(v):
        return g1(u, h, v)

    bound_u = bound
    bound_l = -bound
    bounds = [(bound_l, bound_u)] * l
    result = differential_evolution(solve_V, bounds)
    v = result.x
    return v


# print(fmin_bfgs(g1, np.ones(len(dt['Strike'])) * 0.000001, fprime=g1_prime))


# Call price calculation
def call_price(u, h, V):
    return Ik(h, V, K) * np.exp(-u)


def precision(c_mid, c_model):
    return sklearn.metrics.r2_score(c_mid, c_model)


def curve_arbfree(c_bid, c_ask, c_model):
    check_bid = sum(c_bid > c_model)
    check_ask = sum(c_ask < c_model)
    if check_bid != 0 or check_ask != 0:
        return False
    else:
        return True


def deviation_ratio(u, h, V, dt):
    sum_value = np.sum(np.abs((call_price(u, h, V) - dt['Mid'])) / (dt['Ask'] - dt['Bid']) / len(dt))
    return sum_value


print("Complete")


# %%
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA1.xlsx')
S0 = 918.4
t = mkt_time('2022-01-25', '2022-01-28')
dt = dt[dt["Strike"] / S0 >= 0.7]
dt = dt[dt["Strike"] / S0 <= 1.3]
dt = dt.reset_index(drop=True)

# V = np.array(pd.read_excel('V.xlsx', header = None))[:,0]
K = np.append(0, dt['Strike'])
K = np.append(K, 10000)
V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))*0.000001
u = 0
h = 0

dt['Bid_IV'] = 0
dt['Ask_IV'] = 0
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
sigma = (loss_func(S0, t, dt['Strike'], dt['Mid']) * np.sqrt(t))[0]

# Scaling
dt = dt / S0
sigma = sigma / S0
K = K / S0
S0 = 1

c_bid = dt['Bid']
c_ask = dt['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
# w = np.zeros(len(dt['Strike']))
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]

print("Complete")

# %%
from scipy.optimize import minimize, fsolve, root

V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))

u_l = []
h_l = []
g1_l = []
r_l = []
start_time0 = time.time()
for i in range(0, 5000):
    start_time = time.time()

    # Update u
    u = float(sym.log(cal_Iu1(V, h)))
    u_l.append(u)

    # Update h
    h = solve_Ih1(V)[0]
    h_l.append(h)

    # Update V
    def min_g1(V):
        return g1(u, h, V)

    res = minimize(min_g1, V, method='L-BFGS-B')
    V = res.x

    # def min_g1partial(V):
    #     return g1_partial1(V, u, h)
    # V = fsolve(min_g1partial, V)

    func_val = g1(u, h, V)
    g1_l.append(func_val)

    r = deviation_ratio(u, h, V, dt)
    r_l.append(r)

    print("Iteration %s --- %s seconds ---" % (i + 1, time.time() - start_time))
    print("G1 = ", func_val)
    print("u = %s , h = %s" % (u, h))
    print("Deviation ratio = ", r)

    # Bid/Ask test
    c_model = call_price(u, h, V)
    if curve_arbfree(c_bid, c_ask, c_model):
        print("Arb-free at %s iteration" % (i + 1))
        break

    # Deviation Ratio Test
    if r <= 0.05:
        print("Satisfy fitness at %s iteration" % (i + 1))
        break

    # Convergence Test
    if (i >= 1) and (abs(g1_l[i] - g1_l[i - 1]) <= 10 ** (-10)):
        print("Convergence at %s iteration" % (i + 1))
        break
    # print(np.round(V, 6))

print("Total time is %s seconds ---" % (time.time() - start_time0))


# %% Quasi Newton
D = np.eye(len(dt['Strike']))
a = 0.001
l = []
start_time = time.time()
print(g1_partial(V, u, h))
epsilon = sp_norm(g1_partial(V, u, h))
i = 0

for i in range(0, 10000):
    # while epsilon > 0.001:
    g1_1 = g1_partial(V, u, h)
    d = -D @ np.transpose([g1_1])
    s = a * d
    V_update = V + s.T[0]
    g1_2 = g1_partial(V_update, u, h)
    epsilon = sp_norm(g1_2)
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
    # print(g1_partial(V_update, u, h))
# print(V)
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(l)
plt.show()

# %% Differential Evolution
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/AMZNtest2-2.xlsx')
# V = np.array(pd.read_excel('V.xlsx', header = None))[:,0]
K = np.append(0, dt['Strike'])
K = np.append(K, 10000)
V = np.zeros(len(dt['Strike']))
# V = np.ones(len(dt['Strike']))*0.000001
u = 0
h = 0
S0 = 1
t = mkt_time('2022-03-08', '2022-04-14')
dt['Bid_IV'] = 0
dt['Ask_IV'] = 0
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
dt['Moneyness'] = 0
# sigma = (loss_func(S0, t, dt['Strike'], dt['Mid']) * np.sqrt(t))[0]
sigma = 365.43017578125 / 2720.29

c_bid = dt['Bid']
c_ask = dt['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]

u_l = []
h_l = []
g1_l = []
for i in range(0, 1):
    start_time = time.time()
    u = sym.log(cal_Iu1(V, h))
    u_l.append(u)
    h = solve_Ih(V)
    h_l.append(h)
    V = V_update(u, h, V, 1)
    func_val = g1(u, h, V)
    g1_l.append(func_val)
    # print(V)

    print("Iteration %s --- %s seconds ---" % (i + 1, time.time() - start_time))
    print("G1 = ", func_val)
    print("u = %s , h = %s" % (u, h))
    print(np.round(V, 6))

# %%
start_time = time.time()
g1(u, h, V)
print("--- %s seconds ---" % (time.time() - start_time))

# %% Iteration Visualization
plt.plot(r_l)
plt.legend("r")
plt.xlabel('Iteration')
plt.ylabel('Deviation Ratio')
plt.title("TSLA1, Mat = 3 days")
plt.show()

# pd.DataFrame(V).to_excel('DE_V.xlsx')


# %% Model visualization
dt['Model'] = call_price(u, h, V)
# dt['Model1'] = call_price(u1, h1, V1)
# dt['Model2'] = call_price(u2, h2, V2)
# dt['Model3'] = call_price(u3, h3, V3)

for i in range(0, len(dt)):
    dt.loc[i, 'Bid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Bid'])
    dt.loc[i, 'Ask_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Ask'])
    dt.loc[i, 'Mid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Mid'])
    dt.loc[i, 'Model_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model'])
    # dt.loc[i, 'Model1_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model1'])
    # dt.loc[i, 'Model2_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model2'])
    # dt.loc[i, 'Model3_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model3'])

dt1 = dt[~dt['Bid_IV'].isin([0])]
# dt = dt[~dt['Model_IV'].isin([0])]
plt.scatter(dt1['Strike'], dt1['Ask_IV'], c='b', marker='o', s=10)
plt.scatter(dt1['Strike'], dt1['Bid_IV'], c='orange', marker='^', s=10)
plt.plot(dt1['Strike'], dt1['Mid_IV'], c='purple', linewidth=1)
plt.scatter(dt1['Strike'], dt1['Model_IV'], c='crimson', marker='X', s=15)
# plt.scatter(dt1['Strike'], dt1['Model1_IV'], c='crimson', marker='x', s=10)
# plt.scatter(dt1['Strike'], dt1['Model2_IV'], c='limegreen', marker='x', s=10)
# plt.scatter(dt1['Strike'], dt1['Model3_IV'], c='pink', marker='x', s=10)
plt.xlim(1.2, 1.3)
plt.ylim(0.92, 1)
plt.legend(labels=['Ask', 'Bid', 'Mid', 'Model'])
plt.title("TSLA1, Mat = 3 Days")
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.show()


# %% Call Price Visualization
plt.scatter(dt1['Strike'], dt1['Ask'], c='blue', marker='o', s=10)
plt.scatter(dt1['Strike'], dt1['Bid'], c='orange', marker='^', s=10)
plt.plot(dt1['Strike'], dt1['Mid'], c='purple')
plt.legend(labels=['Ask', 'Bid', 'Mid'])
plt.title("TSLA1, Mat = 3 Days")
plt.ylim(0.0015, 0.006)
plt.xlim(1.2, 1.3)
plt.xlabel('Moneyness')
plt.ylabel('Market Call Price')
plt.show()


# %% Normalized Strike
dt = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/T5.xlsx')
dt['Mid'] = (dt['Bid'] + dt['Ask']) / 2
# dt = dt[dt["Strike"] >= 550]
# dt = dt[dt["Strike"] <= 2400]
# dt = dt.reset_index(drop=True)
t = mkt_time('2022-04-08', '3/17/23')
S0 = 1057.26
F = 1087.68
for i in range(0, len(dt)):
    dt.loc[i, 'Mid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Mid'])
iv = 0.636

dt["Normalized"] = np.log(dt["Strike"] / F) / (iv * np.sqrt(t))
dt = dt[dt["Normalized"] >= -1.3]
dt = dt[dt["Normalized"] <= 1.3]
dt = dt.reset_index(drop=True)

print("# of Data = ", len(dt))
print("Maturity = ", t * 252)
print("K = [%s, %s]" % (dt.loc[0, "Strike"], dt.loc[len(dt) - 1, "Strike"]))
print("K/S0 = [%s, %s]" % (round(dt.loc[0, "Strike"] / S0, 1), round(dt.loc[len(dt) - 1, "Strike"] / S0, 1)))
print("Range = [%s, %s]" % (round(dt.loc[0, "Normalized"], 1), round(dt.loc[len(dt) - 1, "Normalized"], 1)))
