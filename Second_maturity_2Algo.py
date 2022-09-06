# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:34:33 2022

@author: lenovo
"""

import pandas as pd 
from scipy import optimize
from scipy.stats import norm
from scipy.special import erf,erfc
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
import time
import sympy as sym
import sklearn.metrics
import seaborn as sns
from scipy.optimize import fmin_bfgs

import os
# os.chdir(r'C:\Users\lenovo\Desktop\Practicum\Code')

# BSM Option Price Calculation
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
    return (len(t)-1) / 252


print("Complete")


# %%
dt1 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA1.xlsx')#,sheet_name='Call')
dt2 = pd.read_excel(r'/Users/binglianluo/Desktop/Spring2022/Practicum/TSLA2.xlsx')#,sheet_name='Call')
S0 = 918.4
t1 = mkt_time('2022-01-25', '2022-01-28')
t2 = mkt_time('2022-01-25', '2022-02-04')
dt1 = dt1[dt1['Strike']/S0>=0.7]
dt1 = dt1[dt1['Strike']/S0<=1.3]
dt1 = dt1.reset_index(drop = True)
dt2 = dt2[dt2['Strike']/S0>=0.7]
dt2 = dt2[dt2['Strike']/S0<=1.3]
dt2 = dt2.reset_index(drop = True)


K1 = np.append(0, dt1['Strike'])
K1 = np.append(K1, 10000)
K2 = np.append(0, dt2['Strike'])
K2 = np.append(K2, 10000)

dt1['Mid'] = (dt1['Bid'] + dt1['Ask']) / 2
dt2['Mid'] = (dt2['Bid'] + dt2['Ask']) / 2

dt1['Bid_IV'] = 0
dt1['Ask_IV'] = 0
dt1['Moneyness'] = 0
sigmat1 = (loss_func(S0, t1, dt1['Strike'], dt1['Mid']) * np.sqrt(t1))[0]

# Scaling S1
S0 = 918.4
dt1 = dt1 / S0
sigmat1 = sigmat1 / S0
K1 = K1 / S0
S0 = 1

# Scaling S2
S0 = 918.4
dt2 = dt2 / S0
K2 = K2 / S0
S0 = 1

#%%Getting S1 s

def error_func(sig, kmin, n):
    # S: constant spot price
    # T: constant and calculated
    # K: list, imported
    # c: list, imported
    
    def E_func(a, sigma, Kmin, n):
        b = 2 * S0 - a
        A1 = 1 / (2 * sigma ** 2)
        m = S0 - Kmin
        err1=(b-a)/n
        err2=-1/(2*np.sqrt(2*np.pi)*sigma*A1)*np.exp(-A1*(b-S0)**2)
        err3=m/(2*np.sqrt(2*np.pi)*sigma*A1)*erfc(np.sqrt(A1)*(b-S0))
        return (err1+err2+err3)
    a = kmin
    def find_a(x):
        e = E_func(x , sig , kmin, n)
        return e
    ini = kmin

    return optimize.minimize(find_a, ini)

# n = 1
# result = error_func(sigmat1 , dt1['Strike'][0], n)
# a = result.x[0]

#method: paper's a and b

def get_discrete_manual(n):
    s_l=[]
    for i in range(len(dt1)):
        if i%10==0:
            s_l.append(dt1['Strike'].iloc[i])
    return(s_l)

S1_list=get_discrete_manual(10)

def get_discrete(a,n):
    b=2*S0-a
    s_l=np.linspace(a,b,n)
    return s_l
# a=918.4/918.4
n=50
a = 0.5
S1_list=get_discrete(a,n)

def get_weight(s_l,mean,vol):
    denom = np.sum(norm.pd(s_l, mean, vol))
    numer = norm.pd(s_l, mean, vol)
    return (numer/denom )

weight_list = get_weight(S1_list, S0, sigmat1)


#%%
# B* func Calculation
def Bfunc(S, K, t, sigma):
    inside = (K - S) / (np.sqrt(2 * t) * sigma)
    outsideup = sigma * np.sqrt(t) * np.exp(-(K - S) ** 2 / (2 * t * sigma ** 2))
    outsidedown = np.sqrt(2 * np.pi)
    return 0.5 * (S - K) * (1 - erf(inside)) + outsideup / outsidedown


# Part 3.1: Optimization over sigma0
def loss_func_list(s_l, w_l, t, k, c):
    # S: list, price
    # T: constant and calculated
    # K: list, imported
    # c: list, imported
    def find_sigma(x):
        # x: [sigma0 , beta]
        #sig: list
        sig = x[0] * s_l ** x[1]
        a = 0
        for i in range(len(k)):
            expectation = 0
            for j in range(len(s_l)):
                expectation += Bfunc(s_l[j], k[i], t, sig[j])*w_l[j]
            a += (expectation - c[i]) ** 2
        return a
    
    ini = [0.5,0.5]
    return optimize.minimize(find_sigma, ini)

X=loss_func_list(S1_list,weight_list,(t2-t1),dt2['Strike'],dt2['Mid'])

sigmat1_list = X.x[0] * S1_list ** X.x[1] * np.sqrt(t2-t1)

X.x[0] * 1 ** X.x[1] * np.sqrt(t2-t1)
#%%
###################################################################
###################################################################
V=np.zeros(len(dt2))
u=0;h=0;

#%%
# Iu Calculation
def new_cal_Iu(V, h, S0):
    A = (K[0] - S0) / sigma
    B = (K[1] - S0) / sigma
    sum = 0.5 * np.exp((-h * sigma) ** 2 / 2) * \
          (erf((B + h * sigma) / np.sqrt(2)) - erf((A + h * sigma) / np.sqrt(2)))
    for k in range(0, len(dt['Strike'])):
        A = (K[k + 1] - S0) / sigma
        B = (K[k + 2] - S0) / sigma
        alpha = -np.sum(V[0:k + 1]) - h
        beta = np.sum((V[0:k + 1] * K[1:k + 2])) + h * S0

        sum += 0.5 * np.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0) * \
               (erf((B - alpha * sigma) / np.sqrt(2)) - erf((A - alpha * sigma) / np.sqrt(2)))
        # print(alpha, beta, beta + (alpha * sigma) ** 2 / 2 + alpha * S0,
        #       np.exp(beta + (alpha * sigma) ** 2 / 2 + alpha * S0),
        #       (B - alpha * sigma) / np.sqrt(2), (A - alpha * sigma) / np.sqrt(2), sum)
    return sum
def expec_cal_Iu(V,h):
    Iu_sum=0
    for i in range(len(S1_list)):
        Iu_sum += new_cal_Iu(V, h, sigmat1_list[i], K2, dt2, S1_list[i])*weight_list[i]
    return Iu_sum

def new_cal_Iu1(V, h, S0):
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
def expec_cal_Iu1(V,h):
    Iu_sum=0
    for i in range(len(S1_list)):
        Iu_sum += new_cal_Iu1(V, h, sigmat1_list[i], K2, dt2, S1_list[i])*weight_list[i]
    return Iu_sum

#%%
# Ih Calculation
# Solve for h
# Without using the value of u
def new_solve_Ih(V,S0):
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


def new_solve_Ih1(V,S0):
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


#%%
# Ik Calculation
def new_Ik(h, V, S0):
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


def new_Ik1(h, V, S0):
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


#%%
c_bid = dt2['Bid']
c_ask = dt2['Ask']
c_mid = [(a + b) / 2 for a, b in zip(c_bid, c_ask)]
delta_bid = [(a - b) for a, b in zip(c_bid, c_mid)]
delta_ask = [(a - b) for a, b in zip(c_ask, c_mid)]
w = [0.1 * (a - b) for a, b in zip(c_ask, c_bid)]
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
    res = u + sum_f1(V) + np.sum(V * c_mid) + new_cal_Iu(V, h, S0) * np.exp(float(-u))
    return res
def g11(u, h, V):
    res = u + sum_f1(V) + np.sum(V * c_mid) + new_cal_Iu1(V, h) * np.exp(float(-u))
    return res

def new_g1(u_l,h_l,V):
    #u_l,h_l list
    one = np.sum(u_l * weight_list)
    two = sum_f1(V)
    three = np.sum(V * c_mid)
    four = 0
    for s in range(len(S1_list)):
        four += weight_list[s]*np.exp(float(-u_l[s]))*float(new_cal_Iu(V,h_l[s],S1_list[s]))
    return one+two+three+four

def call_price(u, h, V):
    return new_Ik(h, V, S0)*np.exp(-u)

#%%
from scipy.optimize import minimize, fsolve, root

def min_g1(V):
    return g1(u, h, V)
def new_min_g1(V):
    return new_g1(u_l, h_l, V)
def deviation_ratio(u,h,V):
    sum_value=np.sum(np.abs((call_price(u, h, V)-dt['Mid']))/(dt['Ask']-dt['Bid'])/len(dt))
    return sum_value

#Iu_list = []
#Ik_list = []
#g1_l = []
#r_l = []
D_l = []
#u_l=[];h_l=[];V_l=[]
#%%
K=K2;dt=dt2;t=t2-t1;

total_start_time = time.time()

V=np.zeros(len(dt2))

#u=0;h=0;

u_l=np.zeros(len(S1_list));h_l=np.zeros(len(S1_list))

#%%
for i in range(10):
    start_time = time.time()
    for s in range(len(S1_list)):
        # print(s)
        sigma=sigmat1_list[s];S0=S1_list[s]
        u_l[s] = float(sym.log(new_cal_Iu(V, h_l[s], S1_list[s])))
        # print(u_l[s])
        h_l[s] = new_solve_Ih1(V, S1_list[s])[0]
        # print(h_l[s])
    S0=1
    #u=np.average(u_l);h=np.average(h_l)

    print(new_g1(u_l, h_l, V))

    res = minimize(new_min_g1, V, method='L-BFGS-B')

    V = res.x
    # print(V)

    #func_val = g11(u, h, V)

    #g1_l.append(func_val)

    deviation_val = deviation_ratio(u,h,V)

    #if deviation_val<0.05:

        #break

    print("Iteration %s --- %s seconds ---" % (i + 1, time.time() - start_time))

    #print("deviation = %s"% (deviation_val))

        #print("Deviation Ratio = ", deviation_val)

print("ToTal Iteration %s --- %s seconds ---" % (i + 1, time.time() - total_start_time))


#%%
Ik = np.zeros(len(S1_list))
Iu = np.zeros(len(S1_list))
c = np.zeros(len(V))
for s in range(0, len(S1_list)):
    c += weight_list[s] * new_Ik(h_l[s], V, S1_list[s]) / new_cal_Iu(V, h_l[s], S1_list[s])
    # Iu[s] = new_cal_Iu(V, h_l[s], S1_list[s])


#%%
def new_call_price(Iu_list,Ik_list,weight_list):
    #Iu:num_S1 list
    #Ik:num_S1 * num_K array
    Ik_matrix=np.array(Ik_list)
    num_S1,num_K=Ik_matrix.shape
    call_price=np.zeros(num_K)
    for i in range(num_K):
        for j in range(num_S1):
            call_price[i] += Ik_matrix[j,i]/Iu_list[j]*weight_list[j]
    return call_price

#%%
def deviation_ratio(u,h,V):
    sum_value=np.sum(np.abs((call_price(u, h, V)-dt['Mid']))/(dt['Ask']-dt['Bid'])/len(dt))
    return sum_value

#%%

def ivplot(u,h,V,n):
    #meth='mim G1'
    dt['Model'] = new_Ik(h, V, K)*np.exp(-u)
    S0=1
    #dt['Model'] = new_call_price(Iu_list,Ik_list,weight_list)
    for i in range(0, len(dt)):
        dt.loc[i, 'Moneyness'] = dt.loc[i, 'Strike'] / S0
        dt.loc[i, 'Bid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Bid'])
        dt.loc[i, 'Ask_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Ask'])
        dt.loc[i, 'Mid_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Mid'])
        dt.loc[i, 'Model_IV'] = iv_cal('c', S0, dt.loc[i, 'Strike'], t, 0, dt.loc[i, 'Model'])


    dt1 = dt[~dt['Bid_IV'].isin([0])]
    # dt = dt[~dt['Model_IV'].isin([0])]
    plt.scatter(dt1['Moneyness'], dt1['Ask_IV'], c='blue', marker='o', s=10)
    plt.scatter(dt1['Moneyness'], dt1['Bid_IV'], c='orange', marker='^', s=10)
    plt.plot(dt1['Moneyness'], dt1['Mid_IV'], c='purple')
    plt.scatter(dt1['Moneyness'], dt1['Model_IV'], c='red', marker='x', s=10)
    # plt.xlim(0.85, 1.2)
    #plt.ylim(0.5, 0.7)
    plt.legend(labels=['Mid', 'Ask', 'Bid', 'Model'])
    #plt.title("TSLA, Mat = 3-6 Days, Meth=P*, S1( %s )" %(n+1))
    plt.title("TSLA, Mat = 3-6 Days, Meth=P*, AVG") 
    plt.show()

