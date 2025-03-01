import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

N = 177
B = 30


class OptimizationCallbackVariance:
    def __init__(self):
        self.start_time = time.time()
        self.best_obj_val = None
        self.best_time = None

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if self.best_obj_val is None or obj_val < self.best_obj_val:
                self.best_obj_val = obj_val
                self.best_time = time.time() - self.start_time


class OptimizationCallbackExpectedValue:
    def __init__(self):
        self.start_time = time.time()
        self.best_obj_val = 0
        self.best_time = 0

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if self.best_obj_val is None or obj_val > self.best_obj_val:
                self.best_obj_val = obj_val
                self.best_time = time.time() - self.start_time


def variance_minimization(covariance_matrix):
    model_1 = gp.Model("Variance_Minimization")
    model_1.setParam('OutputFlag', 0)
    a_1 = model_1.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a")
    b_1 = model_1.addVars(N, vtype=GRB.BINARY, name="b")

    objective_1 = gp.quicksum(covariance_matrix[i, j] * a_1[i] * a_1[j]
                              for i in range(N) for j in range(N)
                              if not np.isnan(covariance_matrix[i, j]))
    model_1.setObjective(objective_1, GRB.MINIMIZE)

    model_1.addConstr(gp.quicksum(a_1[i] for i in range(N) if not np.isnan(covariance_matrix[i, i])) == 1, "Sum_a")
    model_1.addConstr(gp.quicksum(b_1[i] for i in range(N) if not np.isnan(covariance_matrix[i, i])) <= B, "Sum_b")

    for i in range(N):
        if np.isnan(covariance_matrix[i, i]):
            model_1.addConstr(a_1[i] == 0, f"a_1[{i}] is zero")
            model_1.addConstr(b_1[i] == 0, f"b_1[{i}] is zero")
        else:
            model_1.addConstr(a_1[i] <= b_1[i], f"a_1[{i}] <= b_1[{i}]")

    callback = OptimizationCallbackVariance()
    model_1.setParam(GRB.Param.TimeLimit, 10)
    model_1.optimize(callback)

    a_1_vals = [a_1[i].X for i in range(N)]
    b_1_vals = [b_1[i].X for i in range(N)]

    return callback.best_obj_val, callback.best_time, a_1_vals, b_1_vals


def expected_value_maximization(windowed_data, covariance_matrix, v):
    expected_values = windowed_data.mean()
    model_2 = gp.Model("ExpectedValue_Maximization")
    model_2.setParam('OutputFlag', 0)
    a_2 = model_2.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a")
    b_2 = model_2.addVars(N, vtype=GRB.BINARY, name="b")

    objective_2 = gp.quicksum(expected_values[i] * a_2[i]
                              for i in range(N)
                              if not np.isnan(expected_values[i]))
    model_2.setObjective(objective_2, GRB.MAXIMIZE)

    var_range = gp.quicksum(covariance_matrix[i, j] * a_2[i] * a_2[j]
                            for i in range(N) for j in range(N)
                            if not np.isnan(covariance_matrix[i, j]))
    model_2.addConstr(var_range <= v, "Variance_constraint")
    model_2.addConstr(gp.quicksum(a_2[i] for i in range(N) if not np.isnan(covariance_matrix[i, i])) == 1, "Sum_a")
    model_2.addConstr(gp.quicksum(b_2[i] for i in range(N) if not np.isnan(covariance_matrix[i, i])) <= B, "Sum_b")

    for i in range(N):
        if np.isnan(covariance_matrix[i, i]):
            model_2.addConstr(a_2[i] == 0, f"a_2[{i}] is zero")
            model_2.addConstr(b_2[i] == 0, f"b_2[{i}] is zero")
        else:
            model_2.addConstr(a_2[i] <= b_2[i], f"a_2[{i}] <= b_2[{i}]")

    callback = OptimizationCallbackExpectedValue()
    model_2.setParam(GRB.Param.TimeLimit, 60)
    model_2.optimize(callback)

    a_2_vals = [a_2[i].X for i in range(N)]
    b_2_vals = [b_2[i].X for i in range(N)]

    return callback.best_obj_val, callback.best_time, a_2_vals, b_2_vals


data_df = pd.read_csv("E:/Uni-Sapienza/Semester-1/Automatic-verification/data_final.csv")

pivot_data = data_df.pivot(index='Time', columns='SensorID', values='Value')
t = len(pivot_data)

for i in range(50, t):
    data_window = pivot_data.iloc[i - 50: i]
    covariance = data_window.cov().values
    result = []
    V, V_best_time, a1_values, b1_values = variance_minimization(covariance)
    try:
        R, R_best_time, a_2_values, b_2_values = expected_value_maximization(data_window, covariance, V)
    except Exception as e:
        expected_value = data_window.mean().fillna(0)
        R = sum([expected_value[i] * a1_values[i] for i in range(N)])
        R_best_time = V_best_time
        a_2_values = a1_values
        b_2_values = b1_values

    result.append({
        'Window_Index': i - 50,
        'Opt_Time_Variance': V_best_time,
        'Opt_Time_Expected_Value': R_best_time,
        'R': R,
        'V': V,
        'a_2': a_2_values,
        'b_2': b_2_values
    })
    print(i)
    results_df = pd.DataFrame(result)
    results_df.to_csv('./optimization_results.csv', mode='a', header=False, index=False)




