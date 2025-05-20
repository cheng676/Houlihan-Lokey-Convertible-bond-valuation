import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd
import scipy.stats as stats
import warnings
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


# 在一个已排序的列表 lst 中找到 target 所在的区间上下界值,用来插值算法中计算某个值的上下界（如利率区间的线性插值）
def find_between(rlist, Year):
    rlist = np.array(rlist)
    
    if Year <= rlist.min():  # Year 比所有 rlist 值都小
        return rlist[0], rlist[0]
    elif Year >= rlist.max():  # Year 比所有 rlist 值都大
        return rlist[-1], rlist[-1]

    # 正常查找最近的上下边界
    rlow = rlist[rlist <= Year].max()
    rhigh = rlist[rlist >= Year].min()

    return rlow, rhigh  


@jit(nopython=True)
def explicit_FD(
    principal, time_to_maturity, volatility, risk_free_rate, dividend_yield,
    credit_spread, conversion_price, coupon_rate, coupon_frequency, 
    time_factor_coupon, stock_grid_steps, time_grid_steps, 
    call_price, put_price, S
):
    """
    principal (float): Bond face value.
    time_to_maturity (float): Time until maturity (T).
    volatility (float): Stock price volatility (sigma).
    risk_free_rate (float): Risk-free interest rate (r).
    dividend_yield (float): Dividend yield (d).
    credit_spread (float): Credit spread (rc).
    conversion_price (float): Conversion price (cv).
    coupon_rate (float): Coupon rate (cp).
    coupon_frequency (float): Coupon payment frequency (cfq).
    time_factor_coupon (float): Time factor for coupon payment (tfc).
    stock_grid_steps (int): Number of stock price steps (M).
    time_grid_steps (int): Number of time steps (N).
    call_price (float): Call provision price.
    put_price (float): Put provision price.
    current_stock_price (float): Current stock price (S).
    """
    
    # initialization of grid
    dx = math.log(6 * conversion_price) / stock_grid_steps      
    dt = time_to_maturity / time_grid_steps  
    U_grid = np.zeros((stock_grid_steps + 1, time_grid_steps + 1))
    V_grid = np.zeros((stock_grid_steps + 1, time_grid_steps + 1))

    # Cash Flow(coupon)
    cp_period = int(time_grid_steps / (coupon_frequency * time_to_maturity))
    cp_array = np.arange(cp_period + int(time_factor_coupon * time_grid_steps), time_grid_steps + 1, cp_period)
    last_cp_date = 100000
    cp_at_T = 0
    if time_grid_steps - last_cp_date <= 1:
        cp_at_T = principal * coupon_rate / coupon_frequency
    else:
        cp_at_T = (time_grid_steps - last_cp_date) * principal * coupon_rate / (coupon_frequency * cp_period)

    # Boundary conditions
    for i in range(1, stock_grid_steps):
        if math.exp(i * dx) >= conversion_price:
            U_grid[i, time_grid_steps] = principal * math.exp(i * dx) / conversion_price + cp_at_T
            V_grid[i, time_grid_steps] = cp_at_T
        else:
            U_grid[i, time_grid_steps] = principal + cp_at_T
            V_grid[i, time_grid_steps] = principal + cp_at_T

    # final value
    U_grid[0, time_grid_steps] = principal
    V_grid[0, time_grid_steps] = principal

    for t in range(1, time_grid_steps):
        U_grid[0, t] = 100 / (1 + (risk_free_rate + credit_spread) * dt) ** (time_grid_steps - t)
        V_grid[0, t] = 100 / (1 + (risk_free_rate + credit_spread) * dt) ** (time_grid_steps - t)

    # Backward deduction
    for j in range(time_grid_steps - 1, -1, -1):
        for i in range(1, stock_grid_steps):
            p1 = dt * ((volatility**2 / 2) * (U_grid[i+1, j+1] - 2 * U_grid[i, j+1] + U_grid[i-1, j+1]) / dx**2)
            p2 = dt * (risk_free_rate-dividend_yield-volatility**2 / 2) * (U_grid[i+1, j+1] - U_grid[i-1, j+1]) / (2 * dx)
            p3 = dt * (-risk_free_rate * (U_grid[i, j+1] - V_grid[i, j+1]) - (risk_free_rate + credit_spread) * V_grid[i, j+1])
            aS = (principal + (j % cp_period) * (principal * coupon_rate / 2) / cp_period) * math.exp(i * dx) / conversion_price
            ft = 0
            for k in cp_array: # if there's a coupon payment happening at this time
                if j <= k < j + 1:
                    ft = principal * coupon_rate / 2

            U_grid[i, j] = U_grid[i, j+1] + p1 + p2 + p3 + ft
            H = U_grid[i, j]  # holding value

            #if holding value < conversion value, convert
            U_grid[i, j] = max(H, aS)
            
            if U_grid[i, j] == H:  # no conversion
                q1 = dt * ((volatility**2 / 2) * (V_grid[i+1, j+1] - 2 * V_grid[i, j+1] + V_grid[i-1, j+1]) / dx**2)
                q2 = dt * (risk_free_rate - dividend_yield-volatility**2 / 2) * (V_grid[i+1, j+1] - V_grid[i-1, j+1]) / (2 * dx)
                q3 = dt * (-(risk_free_rate + credit_spread) * V_grid[i, j+1])
                # update V_grid
                V_grid[i, j] = V_grid[i, j+1] + q1 + q2 + q3 + ft
            elif U_grid[i, j] == aS: # conversion happened
                V_grid[i, j] = 0
            
            # take call provision into account
            if call_price > 0:
                if U_grid[i,j] >= call_price:
                    U_grid[i, j] = call_price  # call: bond price =  Call price
                    V_grid[i, j] = 0  # cash = 0

            # Put provision
            if put_price > 0:
                if H <= put_price:
                    U_grid[i, j] = put_price  # put
                    V_grid[i, j] = put_price  # cash = put price

    S_rounded = round(math.log(S) / (math.log(6 * conversion_price) / 225))
    result = U_grid[S_rounded, 0]
    return result
    #return U_grid


# here is crank-nicolson

@jit(nopython=True)
def block_thomas_solve(A_sub, A_diag, A_sup, b_vec):
    n_int = A_diag.shape[0]
    for i in range(n_int - 1):
        diag_inv = np.linalg.inv(A_diag[i])
        M_i = A_sub[i+1].dot(diag_inv)
        A_diag[i+1] -= M_i.dot(A_sup[i])
        b_vec[i+1] -= M_i.dot(b_vec[i])
    diag_inv = np.linalg.inv(A_diag[n_int - 1])
    X = np.zeros((n_int, 2))
    X[n_int - 1] = diag_inv.dot(b_vec[n_int - 1])
    for i in range(n_int - 2, -1, -1):
        tmp = b_vec[i] - A_sup[i].dot(X[i+1])
        diag_inv = np.linalg.inv(A_diag[i])
        X[i] = diag_inv.dot(tmp)
    return X

@jit(nopython=True)
def crank_nicolson_TF_dxdt(Pr, T, sigma, r, d, rc, cv, cp, cfq, tfc, dx, dt, S):
    S_max = 6*cv
    M = max(1, int(math.log(S_max) / dx))
    N = max(1, int(T / dt))
    actual_dx = math.log(S_max) / M
    actual_dt = T / N
    U_grid = np.zeros((M+1, N+1))
    V_grid = np.zeros((M+1, N+1))

    if cfq * T > 0:
        cp_period = int(N / (cfq * T))
    else:
        cp_period = 0
    cp_offset = 0
    if cp_period > 0:
        cp_offset = int(tfc * cp_period)
    if cp_period > 0:
        cp_array = np.arange(cp_period + cp_offset, N+1, cp_period, dtype=np.int64)
    else:
        cp_array = np.empty(0, dtype=np.int64)

    last_cp_date = 100000
    if cp_period <= 0 or (N - last_cp_date) <= 1:
        cp_at_T = Pr * cp / cfq
    else:
        cp_at_T = (N - last_cp_date) * Pr * cp / (cfq * cp_period)

    for i in range(1, M):
        S_i = math.exp(i * actual_dx)
        if S_i >= cv:
            U_grid[i, N] = Pr * S_i / cv + cp_at_T
            V_grid[i, N] = 0.0
        else:
            U_grid[i, N] = Pr
            V_grid[i, N] = Pr
    U_grid[0, N] = Pr
    V_grid[0, N] = Pr
    U_grid[M, N] = Pr * math.exp(M * actual_dx) / cv + cp_at_T
    V_grid[M, N] = 0.0

    for j in range(1, N):
        discount_factor = (1 + (r + rc) * actual_dt) ** (N - j)
        U_grid[0, j] = Pr / discount_factor
        V_grid[0, j] = Pr / discount_factor

    D = sigma**2 / 2.0
    drift = r - d - sigma**2 / 2.0
    a_coef = D / actual_dx**2 - drift / (2 * actual_dx)
    c_coef = D / actual_dx**2 + drift / (2 * actual_dx)
    b_U = -2 * D / actual_dx**2 - r
    b_V = -2 * D / actual_dx**2 - (r + rc)

    n_int = M - 1
    half_dt = actual_dt / 2

    A_sub = np.zeros((n_int, 2, 2))
    A_diag = np.zeros((n_int, 2, 2))
    A_sup = np.zeros((n_int, 2, 2))
    b_vec = np.zeros((n_int, 2))

    diag_block = np.array([[1 - half_dt * b_U, -half_dt * r],
                           [0, 1 - half_dt * b_V]])
    off_block_lower = np.array([[-half_dt * a_coef, 0],
                                [0, -half_dt * a_coef]])
    off_block_upper = np.array([[-half_dt * c_coef, 0],
                                [0, -half_dt * c_coef]])

    diag_block_rhs = np.array([[1 + half_dt * b_U, half_dt * r],
                               [0, 1 + half_dt * b_V]])
    off_block_lower_rhs = np.array([[half_dt * a_coef, 0],
                                    [0, half_dt * a_coef]])
    off_block_upper_rhs = np.array([[half_dt * c_coef, 0],
                                    [0, half_dt * c_coef]])

    X = np.zeros((n_int, 2))
    for i in range(n_int):
        idx = i + 1
        X[i, 0] = U_grid[idx, N]
        X[i, 1] = V_grid[idx, N]

    for n_ in range(N-1, -1, -1):
        for i in range(n_int):
            A_diag[i] = diag_block.copy()
            A_sub[i]  = off_block_lower.copy()
            A_sup[i]  = off_block_upper.copy()

        for i in range(n_int):
            tmp = diag_block_rhs.dot(X[i])
            if i > 0:
                tmp += off_block_lower_rhs.dot(X[i-1])
            if i < n_int - 1:
                tmp += off_block_upper_rhs.dot(X[i+1])
            b_vec[i] = tmp

        # boundary
        left_bc = np.array([U_grid[0, n_+1], V_grid[0, n_+1]])
        b_vec[0] -= off_block_lower_rhs.dot(left_bc)
        right_bc = np.array([U_grid[M, n_+1], V_grid[M, n_+1]])
        b_vec[n_int - 1] -= off_block_upper_rhs.dot(right_bc)

        # coupon
        ft = 0.0
        for k_ in cp_array:
            if n_ <= k_ < n_+1:
                ft = Pr * cp / cfq
                break
        for i in range(n_int):
            b_vec[i, 0] += ft

        X_new = block_thomas_solve(A_sub, A_diag, A_sup, b_vec)
        X = X_new.copy()
        for i in range(n_int):
            idx = i + 1
            U_grid[idx, n_] = X[i, 0]
            V_grid[idx, n_] = X[i, 1]

        # early exercise
        for i in range(1, M):
            S_i = math.exp(i * actual_dx)
            accrual = 0.0
            if cp_period > 0:
                accrual = (((N - n_) % cp_period) * (Pr * cp / cfq)) / cp_period
            conv_val = (Pr + accrual) * S_i / cv
            if U_grid[i, n_] < conv_val:
                U_grid[i, n_] = conv_val
                V_grid[i, n_] = 0.0

        discount_factor = (1 + (r + rc) * actual_dt) ** (N - n_)
        U_grid[0, n_] = Pr / discount_factor
        V_grid[0, n_] = Pr / discount_factor
        U_grid[M, n_] = Pr * math.exp(M * actual_dx) / cv
        V_grid[M, n_] = 0.0

    actual_dx = math.log(450.0) / M
    S_app = round(math.log(S) / actual_dx)
    S_app = max(0, min(S_app, M))
    estimated_price = U_grid[S_app, 0]
    return estimated_price