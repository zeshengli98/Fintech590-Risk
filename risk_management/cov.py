import numpy as np
import pandas as pd

'''
For estimating the weighted cov in this method:
df: the dataframe format, demanding the time series is ascending
'''
def weighted_pair(x, y, weight):
    n = len(weight)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = 0
    for i in range(n):
        cov += weight[n - 1 - i]*(x[i] - mean_x) * (y[i] - mean_y)
    return cov

def calculate_weight(lamb, df):
    X = df.index.values
    weight = [(1-lamb)*lamb**(i-1) for i in X]
    weight_adjust = [weight[i]/sum(weight) for i in X]
    return weight_adjust

def weighted_cov(lamb, df):

    n = df.shape[1]
    T = len(df)
    weight = calculate_weight(lamb, df)
    cov_mat = pd.DataFrame(np.zeros((n, n)))
    for i in range(n):
        x = df.iloc[:, i]
        cov_mat.iloc[i, i] = weighted_pair(x, x, weight)
        for j in range(i+1):
            y = df.iloc[:, j]
            cov_mat.iloc[i, j] = weighted_pair(x, y, weight)
            cov_mat.iloc[j, i] = cov_mat.iloc[i, j]

    return np.array(cov_mat)


if __name__== "__main__":
    data = pd.read_csv("../DailyReturn.csv").iloc[:, 1:]

    data2 = pd.read_csv("../problem1.csv")

    print(weighted_cov(0.95,data2))

